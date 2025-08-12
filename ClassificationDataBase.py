import yaml

def get_prompt_by_version(yaml_path: str, version: str) -> dict | None:
    """
    Load a YAML file and return the prompt dictionary for a given PROMPT_VERSION.
    
    :param yaml_path: Path to the YAML file
    :param version: PROMPT_VERSION to search for
    :return: Dict for the matching prompt, or None if not found
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    prompts = data.get("PROMPTS", [])
    
    for prompt in prompts:
        if prompt.get("PROMPT_VERSION") == version:
            return prompt
    
    return None  # No match found

# Example usage:
if __name__ == "__main__":
    yaml_file = "prompts.yaml"
    version_to_find = "v2"
    prompt_dict = get_prompt_by_version(yaml_file, version_to_find)

    if prompt_dict:
        print(f"Found prompt for version '{version_to_find}':")
        print(prompt_dict)
    else:
        print(f"No prompt found for version '{version_to_find}'.")
