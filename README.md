# pi-supergrok

Pi extension that adds SuperGrok/xAI OAuth for Grok models.

Supported models:

• grok-4.20-0309-non-reasoning
• grok-4.20-0309-reasoning
• grok-4.3
• grok-build-0.1
• grok-composer-2.5-fast

Requires pi `0.74.0` or newer.

## Install

```bash
pi install npm:pi-supergrok
```

Or try without installing:

```bash
pi -e npm:pi-supergrok
```

## Usage

Reload pi or restart it, then authenticate one of the registered providers:

```text
/login supergrok
```

Then select a model such as:

```text
/model supergrok/grok-4.3
```

## Uninstall

```bash
pi remove npm:pi-supergrok
```

## License

MIT
