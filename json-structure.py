import json

def loadAndPrintJson(json_path):
    with open(json_path) as f:
        data = json.load(f)

    def printKeysRecursive(obj, indent=0):
        indent_str = "  " * indent
        if isinstance(obj, dict):
            print(indent_str + "{")
            for k, v in obj.items():
                print(f"{indent_str}  {k}: ", end="")
                printKeysRecursive(v, indent + 1)
            print(indent_str + "}")
        elif isinstance(obj, list):
            # If it's an array of arrays of numbers at the bottom, simplify
            if all(isinstance(x, list) and
                  all(isinstance(y, (int, float)) for y in x)
                  for x in obj):
                print("[[int]]")
                return
            # Otherwise print normally
            print("[")
            for item in obj:
                print(indent_str + "  ", end="")
                printKeysRecursive(item, indent + 1)
            print(indent_str + "]")
        else:
            print("[" + type(obj).__name__ + "]")

    printKeysRecursive(data)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Please provide a JSON file path")
        sys.exit(1)
    loadAndPrintJson(sys.argv[1])
