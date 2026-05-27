# pi-supergrok

Pi extension/package that adds SuperGrok/xAI OAuth for Grok models.

## Usage

Reload pi or restart it, then authenticate one of the registered providers:

```text
/login xai
```

or:

```text
/login supergrok
```

Then select a model such as:

```text
/model xai/grok-build-0.1
/model supergrok/grok-4.3
```

`xai` is pi's built-in API-key provider. `supergrok` is this extension's OAuth-only provider with current Grok OAuth model IDs.
