from araras.kernel.monitoring import print_warning_message
def main():
    print("Testing print_warning_message from araras.kernel.monitoring")
    try:
        res = print_warning_message("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
