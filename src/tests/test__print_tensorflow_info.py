from araras.utils.gpu import _print_tensorflow_info
def main():
    print("Testing _print_tensorflow_info from araras.utils.gpu")
    try:
        res = _print_tensorflow_info()
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
