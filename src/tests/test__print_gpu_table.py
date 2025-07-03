from araras.utils.gpu import _print_gpu_table
def main():
    print("Testing _print_gpu_table from araras.utils.gpu")
    try:
        res = _print_gpu_table("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
