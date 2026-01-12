import re


def render_template(template: str, context: dict) -> str:
    # Replace variables with their values from the context dictionary
    template = re.sub(
        r"{{\s*(\w+)\s*}}", lambda m: context.get(m.group(1), m.group()), template
    )

    # Process conditional blocks
    def process_if(match):
        condition, text = match.groups()
        if bool(context.get(condition, False)):
            return text.strip()
        else:
            return ""

    while True:
        new_template = re.sub(
            r"{%\s*if\s+(\w+)\s*%}(.*?){%\s*endif\s*%}", process_if, template
        )
        if new_template == template:
            break
        template = new_template

    return template
