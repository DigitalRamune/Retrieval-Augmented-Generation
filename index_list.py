import pathlib as p


def get_files_for_index() -> list[str]:
    d = p.Path.cwd()
    python_files = d.rglob('*.py')
    md_files = d.rglob('*.md')
    final_list = []
    for file in python_files:
        final_list.append(file)
    for file in md_files:
        final_list.append(file)
    return final_list

if __name__ == "__main__":
    list = get_files_for_index("./vllm-0.10.1/")
    for item in list:
        print(f'{item}\n')