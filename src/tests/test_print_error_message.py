from araras.kernel.monitoring import print_error_message
def main():
    print("Testing print_error_message from araras.kernel.monitoring")
    try:
        res = print_error_message("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
