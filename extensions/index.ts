import crypto from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import http, { type IncomingMessage, type ServerResponse } from "node:http";
import { homedir } from "node:os";
import { join } from "node:path";
import {
	type Api,
	type AssistantMessageEventStream,
	type Context,
	createAssistantMessageEventStream,
	type Model,
	type OAuthCredentials,
	type OAuthLoginCallbacks,
	type SimpleStreamOptions,
	streamSimpleOpenAICompletions,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const XAI_API_BASE_URL = "https://api.x.ai/v1";
const XAI_OAUTH_ISSUER = "https://auth.x.ai";
const XAI_OAUTH_AUTHORIZE_URL = `${XAI_OAUTH_ISSUER}/oauth2/authorize`;
const XAI_OAUTH_DISCOVERY_URL = `${XAI_OAUTH_ISSUER}/.well-known/openid-configuration`;

// Public desktop OAuth client ID used by xAI/Grok CLI style clients. This is not a secret.
const XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828";
const XAI_OAUTH_SCOPE =
	"openid profile email offline_access grok-cli:access api:access";

const REDIRECT_HOST = "127.0.0.1";
const REDIRECT_PORT = 56121;
const REDIRECT_PATH = "/callback";
const CALLBACK_TIMEOUT_MS = 3 * 60 * 1000;
const REFRESH_SKEW_MS = 2 * 60 * 1000;
const REFRESH_PREFIX = "xai:";

interface XaiDiscovery {
	authorizationEndpoint: string;
	tokenEndpoint: string;
}

interface XaiTokenPayload {
	accessToken: string;
	refreshToken: string;
	expiresAt: number;
}

interface XaiRefreshParts {
	refreshToken: string;
	tokenEndpoint?: string;
	redirectUri?: string;
}

interface OAuthListener {
	redirectUri: string;
	waitForCallback(timeoutMs: number): Promise<URL>;
	close(): Promise<void>;
}

type ProviderModelConfig = {
	id: string;
	name: string;
	reasoning: boolean;
	input: ("text" | "image")[];
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
	};
	contextWindow: number;
	maxTokens: number;
};

/**
 * Models that should always be available for the supergrok provider,
 * even if they are not (yet) returned by the live /v1/models endpoint
 * or when the user has not yet run /login.
 *
 * These are merged with dynamically fetched models (live ones are appended
 * if they don't conflict on id).
 */
const STATIC_SUPERGROK_MODELS: ProviderModelConfig[] = [
	{
		id: "grok-composer-2.5-fast",
		name: "Grok Composer 2.5 Fast (SuperGrok)",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 131_072,
		maxTokens: 8192,
	},
];

function generatePkce(): { verifier: string; challenge: string } {
	const verifier = crypto.randomBytes(48).toString("base64url");
	const challenge = crypto
		.createHash("sha256")
		.update(verifier)
		.digest("base64url");
	return { verifier, challenge };
}

function validateXaiOAuthEndpoint(url: string, field = "endpoint"): string {
	const parsed = new URL(url);
	if (parsed.protocol !== "https:") {
		throw new Error(
			`xAI OAuth discovery returned a non-HTTPS ${field}: ${url}`,
		);
	}

	const host = parsed.hostname.toLowerCase();
	if (host !== "x.ai" && !host.endsWith(".x.ai")) {
		throw new Error(
			`xAI OAuth discovery ${field} host ${host} is not on xAI's origin.`,
		);
	}

	return url;
}

async function discoverXaiOAuth(): Promise<XaiDiscovery> {
	const response = await fetch(XAI_OAUTH_DISCOVERY_URL, {
		headers: { Accept: "application/json" },
	});
	if (!response.ok) {
		throw new Error(`xAI OIDC discovery failed with HTTP ${response.status}.`);
	}

	const payload = (await response.json()) as Record<string, unknown>;
	const authorizationEndpoint = String(
		payload.authorization_endpoint ?? "",
	).trim();
	const tokenEndpoint = String(payload.token_endpoint ?? "").trim();
	if (!authorizationEndpoint || !tokenEndpoint) {
		throw new Error(
			"xAI OIDC discovery did not include authorization and token endpoints.",
		);
	}

	return {
		authorizationEndpoint: validateXaiOAuthEndpoint(
			authorizationEndpoint,
			"authorization_endpoint",
		),
		tokenEndpoint: validateXaiOAuthEndpoint(tokenEndpoint, "token_endpoint"),
	};
}

function buildXaiAuthorizeUrl(input: {
	authorizationEndpoint: string;
	redirectUri: string;
	codeChallenge: string;
	state: string;
	nonce: string;
}): string {
	validateXaiOAuthEndpoint(
		input.authorizationEndpoint,
		"authorization_endpoint",
	);

	// The Hermes/SuperGrok flow currently expects /oauth2/authorize and referrer=hermes-agent.
	const url = new URL(XAI_OAUTH_AUTHORIZE_URL);
	url.searchParams.set("response_type", "code");
	url.searchParams.set("client_id", XAI_OAUTH_CLIENT_ID);
	url.searchParams.set("redirect_uri", input.redirectUri);
	url.searchParams.set("scope", XAI_OAUTH_SCOPE);
	url.searchParams.set("code_challenge", input.codeChallenge);
	url.searchParams.set("code_challenge_method", "S256");
	url.searchParams.set("state", input.state);
	url.searchParams.set("nonce", input.nonce);
	url.searchParams.set("plan", "generic");
	url.searchParams.set("referrer", "hermes-agent");
	return url.toString();
}

async function exchangeXaiCodeForTokens(input: {
	tokenEndpoint: string;
	code: string;
	redirectUri: string;
	codeVerifier: string;
	codeChallenge: string;
}): Promise<XaiTokenPayload> {
	const tokenEndpoint = validateXaiOAuthEndpoint(
		input.tokenEndpoint,
		"token_endpoint",
	);
	const startedAt = Date.now();
	const response = await fetch(tokenEndpoint, {
		method: "POST",
		headers: {
			"Content-Type": "application/x-www-form-urlencoded",
			Accept: "application/json",
		},
		body: new URLSearchParams({
			grant_type: "authorization_code",
			code: input.code,
			redirect_uri: input.redirectUri,
			client_id: XAI_OAUTH_CLIENT_ID,
			code_verifier: input.codeVerifier,
			code_challenge: input.codeChallenge,
			code_challenge_method: "S256",
		}),
	});

	return parseTokenResponse(response, startedAt, "xAI token exchange failed");
}

async function refreshXaiTokens(
	credentials: OAuthCredentials,
): Promise<OAuthCredentials> {
	const parts = parseXaiRefresh(credentials.refresh);
	if (!parts.refreshToken) {
		throw new Error("xAI OAuth refresh token is missing. Run /login again.");
	}

	const tokenEndpoint =
		parts.tokenEndpoint ?? (await discoverXaiOAuth()).tokenEndpoint;
	const startedAt = Date.now();
	const response = await fetch(
		validateXaiOAuthEndpoint(tokenEndpoint, "token_endpoint"),
		{
			method: "POST",
			headers: {
				"Content-Type": "application/x-www-form-urlencoded",
				Accept: "application/json",
			},
			body: new URLSearchParams({
				grant_type: "refresh_token",
				client_id: XAI_OAUTH_CLIENT_ID,
				refresh_token: parts.refreshToken,
			}),
		},
	);

	const refreshed = await parseTokenResponse(
		response,
		startedAt,
		"xAI token refresh failed",
		parts.refreshToken,
	);

	return {
		refresh: packXaiRefresh({
			refreshToken: refreshed.refreshToken,
			tokenEndpoint,
			redirectUri: parts.redirectUri,
		}),
		access: refreshed.accessToken,
		expires: refreshed.expiresAt - REFRESH_SKEW_MS,
	};
}

async function parseTokenResponse(
	response: Response,
	startedAt: number,
	errorPrefix: string,
	fallbackRefreshToken = "",
): Promise<XaiTokenPayload> {
	const text = await response.text();
	if (!response.ok) {
		throw new Error(
			`${errorPrefix} (HTTP ${response.status}).${text ? ` Response: ${text}` : ""}`,
		);
	}

	let payload: Record<string, unknown>;
	try {
		payload = JSON.parse(text) as Record<string, unknown>;
	} catch {
		throw new Error(`${errorPrefix}: response was not valid JSON.`);
	}

	const accessToken = String(payload.access_token ?? "").trim();
	const refreshToken = String(
		payload.refresh_token ?? fallbackRefreshToken,
	).trim();
	if (!accessToken)
		throw new Error(`${errorPrefix}: response did not include access_token.`);
	if (!refreshToken)
		throw new Error(`${errorPrefix}: response did not include refresh_token.`);

	return {
		accessToken,
		refreshToken,
		expiresAt: calculateTokenExpiry(startedAt, payload.expires_in, accessToken),
	};
}

function calculateTokenExpiry(
	requestTimeMs: number,
	expiresInSeconds: unknown,
	accessToken?: string,
): number {
	if (
		typeof expiresInSeconds === "number" &&
		Number.isFinite(expiresInSeconds) &&
		expiresInSeconds > 0
	) {
		return requestTimeMs + expiresInSeconds * 1000;
	}

	const jwtExpiry = getJwtExpiry(accessToken);
	return jwtExpiry ?? requestTimeMs + 3600 * 1000;
}

function getJwtExpiry(token?: string): number | undefined {
	if (!token?.includes(".")) return undefined;
	try {
		const payload = token.split(".")[1];
		if (!payload) return undefined;
		const parsed = JSON.parse(
			Buffer.from(payload, "base64url").toString("utf8"),
		) as Record<string, unknown>;
		const exp = parsed.exp;
		return typeof exp === "number" && Number.isFinite(exp)
			? exp * 1000
			: undefined;
	} catch {
		return undefined;
	}
}

function packXaiRefresh(parts: XaiRefreshParts): string {
	return `${REFRESH_PREFIX}${Buffer.from(JSON.stringify(parts), "utf8").toString("base64url")}`;
}

function parseXaiRefresh(refresh: string): XaiRefreshParts {
	const value = (refresh ?? "").trim();
	if (!value) return { refreshToken: "" };
	if (!value.startsWith(REFRESH_PREFIX)) return { refreshToken: value };

	try {
		const parsed = JSON.parse(
			Buffer.from(value.slice(REFRESH_PREFIX.length), "base64url").toString(
				"utf8",
			),
		) as Record<string, unknown>;
		return {
			refreshToken:
				typeof parsed.refreshToken === "string" ? parsed.refreshToken : "",
			tokenEndpoint:
				typeof parsed.tokenEndpoint === "string"
					? parsed.tokenEndpoint
					: undefined,
			redirectUri:
				typeof parsed.redirectUri === "string" ? parsed.redirectUri : undefined,
		};
	} catch {
		return { refreshToken: "" };
	}
}

function parseOAuthCallbackInput(
	input: string,
	expectedState: string,
): { code: string } | { error: string } {
	const raw = input.trim();
	if (!raw) return { error: "Missing authorization code." };

	try {
		const url = new URL(raw);
		const oauthError = url.searchParams.get("error");
		if (oauthError)
			return { error: url.searchParams.get("error_description") ?? oauthError };

		const code = url.searchParams.get("code") ?? "";
		const state = url.searchParams.get("state") ?? "";
		if (!code) return { error: "Missing authorization code in callback." };
		if (state !== expectedState) return { error: "OAuth state mismatch." };
		return { code };
	} catch {
		return raw ? { code: raw } : { error: "Missing authorization code." };
	}
}

async function loginXai(
	callbacks: OAuthLoginCallbacks,
): Promise<OAuthCredentials> {
	const discovery = await discoverXaiOAuth();
	const listener = await startXaiOAuthListener();
	const pkce = generatePkce();
	const state = crypto.randomBytes(24).toString("hex");
	const nonce = crypto.randomBytes(24).toString("hex");
	const authorizationUrl = buildXaiAuthorizeUrl({
		authorizationEndpoint: discovery.authorizationEndpoint,
		redirectUri: listener.redirectUri,
		codeChallenge: pkce.challenge,
		state,
		nonce,
	});

	callbacks.onAuth({ url: authorizationUrl });

	try {
		const callbackUrl = await listener.waitForCallback(CALLBACK_TIMEOUT_MS);
		const params = parseOAuthCallbackInput(callbackUrl.toString(), state);
		if ("error" in params) throw new Error(params.error);

		const tokenPayload = await exchangeXaiCodeForTokens({
			tokenEndpoint: discovery.tokenEndpoint,
			code: params.code,
			redirectUri: listener.redirectUri,
			codeVerifier: pkce.verifier,
			codeChallenge: pkce.challenge,
		});

		return {
			refresh: packXaiRefresh({
				refreshToken: tokenPayload.refreshToken,
				tokenEndpoint: discovery.tokenEndpoint,
				redirectUri: listener.redirectUri,
			}),
			access: tokenPayload.accessToken,
			expires: tokenPayload.expiresAt - REFRESH_SKEW_MS,
		};
	} finally {
		await listener.close().catch(() => undefined);
	}
}

async function startXaiOAuthListener(
	preferredPort = REDIRECT_PORT,
): Promise<OAuthListener> {
	let resolveCallback: ((url: URL) => void) | undefined;
	let _rejectCallback: ((error: Error) => void) | undefined;

	const callbackPromise = new Promise<URL>((resolve, reject) => {
		resolveCallback = resolve;
		_rejectCallback = reject;
	});

	const server = http.createServer((req, res) => {
		handleRequest(req, res, (url) => resolveCallback?.(url));
	});

	const port = await listenWithFallback(server, preferredPort);
	const redirectUri = `http://${REDIRECT_HOST}:${port}${REDIRECT_PATH}`;

	return {
		redirectUri,
		waitForCallback(timeoutMs: number) {
			const timeout = new Promise<never>((_, reject) => {
				setTimeout(
					() =>
						reject(new Error("Timed out waiting for the xAI OAuth callback.")),
					timeoutMs,
				);
			});
			return Promise.race([callbackPromise, timeout]);
		},
		close() {
			return new Promise<void>((resolve) => {
				_rejectCallback = undefined;
				server.close(() => resolve());
			});
		},
	};
}

function listenWithFallback(
	server: http.Server,
	preferredPort: number,
): Promise<number> {
	return new Promise((resolve, reject) => {
		const tryListen = (port: number, allowFallback: boolean) => {
			const onError = (error: NodeJS.ErrnoException) => {
				server.off("listening", onListening);
				if (allowFallback && error.code === "EADDRINUSE") {
					tryListen(0, false);
					return;
				}
				reject(error);
			};

			const onListening = () => {
				server.off("error", onError);
				const address = server.address();
				if (!address || typeof address === "string") {
					reject(new Error("Could not determine xAI OAuth callback port."));
					return;
				}
				resolve(address.port);
			};

			server.once("error", onError);
			server.once("listening", onListening);
			server.listen(port, REDIRECT_HOST);
		};

		tryListen(preferredPort, preferredPort !== 0);
	});
}

const ALLOWED_CALLBACK_ORIGINS = new Set([
	"https://accounts.x.ai",
	"https://auth.x.ai",
]);

function handleRequest(
	req: IncomingMessage,
	res: ServerResponse,
	onCallback: (url: URL) => void,
): void {
	const origin = req.headers.origin;
	const allowOrigin =
		typeof origin === "string" && ALLOWED_CALLBACK_ORIGINS.has(origin)
			? origin
			: "";
	if (allowOrigin) {
		res.setHeader("Access-Control-Allow-Origin", allowOrigin);
		res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
		res.setHeader("Access-Control-Allow-Headers", "Content-Type");
		res.setHeader("Access-Control-Allow-Private-Network", "true");
		res.setHeader("Vary", "Origin");
	}

	if (req.method === "OPTIONS") {
		res.writeHead(204);
		res.end();
		return;
	}

	const host = req.headers.host ?? `${REDIRECT_HOST}:${REDIRECT_PORT}`;
	const url = new URL(req.url ?? "/", `http://${host}`);
	if (req.method !== "GET" || url.pathname !== REDIRECT_PATH) {
		res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
		res.end("Not found.");
		return;
	}

	onCallback(url);
	res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
	const failed = url.searchParams.has("error");
	res.end(
		`<html><body><h1>${failed ? "xAI authorization failed." : "xAI authorization received."}</h1><p>You can close this tab and return to pi.</p></body></html>`,
	);
}

type StoredAuthFile = Record<
	string,
	({ type?: string } & Partial<OAuthCredentials>) | undefined
>;

type XaiModelPayload = {
	data?: Array<Record<string, unknown>>;
};

function authJsonPath(): string {
	return join(homedir(), ".pi", "agent", "auth.json");
}

function readStoredAuth(): StoredAuthFile {
	const path = authJsonPath();
	if (!existsSync(path)) return {};
	return JSON.parse(readFileSync(path, "utf8")) as StoredAuthFile;
}

function writeStoredAuth(auth: StoredAuthFile): void {
	writeFileSync(authJsonPath(), `${JSON.stringify(auth, null, 2)}\n`, "utf8");
}

async function getStoredOAuthCredentials(): Promise<
	OAuthCredentials | undefined
> {
	const auth = readStoredAuth();
	for (const provider of ["supergrok", "xai"] as const) {
		const credentials = auth[provider];
		if (
			credentials?.type !== "oauth" ||
			!credentials.access ||
			!credentials.refresh ||
			!credentials.expires
		) {
			continue;
		}

		const current: OAuthCredentials = {
			access: credentials.access,
			refresh: credentials.refresh,
			expires: credentials.expires,
		};

		if (Date.now() < current.expires) return current;

		const refreshed = await refreshXaiTokens(current);
		auth[provider] = { type: "oauth", ...refreshed };
		writeStoredAuth(auth);
		return refreshed;
	}

	return undefined;
}

async function fetchSuperGrokModels(): Promise<ProviderModelConfig[]> {
	const credentials = await getStoredOAuthCredentials().catch(() => undefined);
	if (!credentials?.access) return [];

	const response = await fetch(`${XAI_API_BASE_URL}/models`, {
		headers: {
			Accept: "application/json",
			Authorization: `Bearer ${credentials.access}`,
			"x-grok-source": "pi-supergrok",
		},
	});

	if (!response.ok) {
		throw new Error(
			`Failed to fetch xAI models: HTTP ${response.status} ${await response.text()}`,
		);
	}

	const payload = (await response.json()) as XaiModelPayload;
	return (payload.data ?? [])
		.map(toProviderModelConfig)
		.filter((model): model is ProviderModelConfig => model !== undefined);
}

function toProviderModelConfig(
	raw: Record<string, unknown>,
): ProviderModelConfig | undefined {
	const id = String(raw.id ?? "").trim();
	if (!id) return undefined;
	// The provider uses Chat Completions; skip non-chat generation models returned by /v1/models.
	// Also skip multi-agent, which is not usable as a normal single chat-completions model in pi.
	if (/image|video|imagine|multi-agent/i.test(id)) return undefined;

	const contextWindow =
		numberFrom(raw, [
			"context_window",
			"contextWindow",
			"max_context_window",
			"maxContextWindow",
			"context_length",
			"contextLength",
		]) ?? 131_072;
	const maxTokens =
		numberFrom(raw, [
			"max_output_tokens",
			"maxOutputTokens",
			"max_tokens",
			"maxTokens",
		]) ?? 8192;
	const supportsImages =
		booleanFrom(raw, [
			"supports_images",
			"supportsImages",
			"vision",
			"image",
		]) ??
		arrayIncludes(raw.input, "image") ??
		arrayIncludes(raw.capabilities, "image") ??
		true;
	const reasoning =
		booleanFrom(raw, [
			"reasoning",
			"supports_reasoning",
			"supportsReasoning",
		]) ?? !/non[-_ ]?reasoning|code[-_ ]?fast|^grok-3(?:-|$)/i.test(id);

	return {
		id,
		name: String(raw.name ?? displayName(id)).trim(),
		reasoning,
		input: supportsImages ? ["text", "image"] : ["text"],
		cost: costFrom(raw),
		contextWindow,
		maxTokens,
	};
}

function numberFrom(
	raw: Record<string, unknown>,
	keys: string[],
): number | undefined {
	for (const key of keys) {
		const value = raw[key];
		if (typeof value === "number" && Number.isFinite(value) && value > 0)
			return value;
		if (
			typeof value === "string" &&
			value.trim() &&
			Number.isFinite(Number(value))
		)
			return Number(value);
	}
	return undefined;
}

function booleanFrom(
	raw: Record<string, unknown>,
	keys: string[],
): boolean | undefined {
	for (const key of keys) {
		const value = raw[key];
		if (typeof value === "boolean") return value;
	}
	return undefined;
}

function arrayIncludes(value: unknown, item: string): boolean | undefined {
	if (!Array.isArray(value)) return undefined;
	return value.some((entry) => String(entry).toLowerCase() === item);
}

function costFrom(raw: Record<string, unknown>): ProviderModelConfig["cost"] {
	const pricing =
		typeof raw.pricing === "object" && raw.pricing
			? (raw.pricing as Record<string, unknown>)
			: raw;
	return {
		input:
			numberFrom(pricing, [
				"input",
				"prompt",
				"input_cost_per_million",
				"prompt_cost_per_million",
			]) ??
			xaiPriceFrom(pricing, "prompt_text_token_price") ??
			0,
		output:
			numberFrom(pricing, [
				"output",
				"completion",
				"output_cost_per_million",
				"completion_cost_per_million",
			]) ??
			xaiPriceFrom(pricing, "completion_text_token_price") ??
			0,
		cacheRead:
			numberFrom(pricing, [
				"cacheRead",
				"cache_read",
				"cache_read_cost_per_million",
			]) ??
			xaiPriceFrom(pricing, "cached_prompt_text_token_price") ??
			0,
		cacheWrite:
			numberFrom(pricing, [
				"cacheWrite",
				"cache_write",
				"cache_write_cost_per_million",
			]) ?? 0,
	};
}

function xaiPriceFrom(
	raw: Record<string, unknown>,
	key: string,
): number | undefined {
	const value = numberFrom(raw, [key]);
	return value === undefined ? undefined : value / 10_000;
}

function displayName(id: string): string {
	return `${id
		.split(/[-_]/g)
		.filter(Boolean)
		.map((part) => part.charAt(0).toUpperCase() + part.slice(1))
		.join(" ")} (SuperGrok)`;
}

function streamSuperGrokWithOAuth(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	(async () => {
		try {
			if (!options?.apiKey) {
				throw new Error(
					"No SuperGrok OAuth token found. Run /login supergrok, then retry.",
				);
			}

			const inner = streamSimpleOpenAICompletions(
				model as Model<"openai-completions">,
				context,
				{
					...options,
					apiKey: options.apiKey,
					headers: { ...options.headers, "x-grok-source": "pi-supergrok" },
				},
			);

			for await (const event of inner) stream.push(event);
			stream.end();
		} catch (error) {
			stream.push({
				type: "error",
				reason: options?.signal?.aborted ? "aborted" : "error",
				error: {
					role: "assistant",
					content: [],
					api: model.api,
					provider: model.provider,
					model: model.id,
					usage: {
						input: 0,
						output: 0,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 0,
						cost: {
							input: 0,
							output: 0,
							cacheRead: 0,
							cacheWrite: 0,
							total: 0,
						},
					},
					stopReason: options?.signal?.aborted ? "aborted" : "error",
					errorMessage: error instanceof Error ? error.message : String(error),
					timestamp: Date.now(),
				},
			});
			stream.end();
		}
	})();

	return stream;
}

const oauth = {
	name: "SuperGrok / xAI OAuth",
	login: loginXai,
	refreshToken: refreshXaiTokens,
	getApiKey: (credentials: OAuthCredentials) => credentials.access,
};

export default async function (pi: ExtensionAPI) {
	const liveModels = await fetchSuperGrokModels();

	// Always include static/special models (e.g. composer variants that may not
	// appear in the public /v1/models response) and append any additional
	// models returned by the authenticated upstream endpoint.
	// Live models with the same id as a static one are ignored (static wins).
	const models: ProviderModelConfig[] = [...STATIC_SUPERGROK_MODELS];
	for (const m of liveModels) {
		if (!models.some((existing) => existing.id === m.id)) {
			models.push(m);
		}
	}

	// Expose SuperGrok separately from pi's built-in xAI API-key provider.
	pi.registerProvider("supergrok", {
		name: "SuperGrok (xAI OAuth)",
		baseUrl: XAI_API_BASE_URL,
		api: "openai-completions",
		headers: { "x-grok-source": "pi-supergrok" },
		oauth,
		streamSimple: streamSuperGrokWithOAuth,
		models,
	});
}
