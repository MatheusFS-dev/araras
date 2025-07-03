from araras.utils.gpu import _print_memory_summary
def main():
    print("Testing _print_memory_summary from araras.utils.gpu")
    try:
        res = _print_memory_summary("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
