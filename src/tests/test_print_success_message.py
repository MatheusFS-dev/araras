from araras.kernel.monitoring import print_success_message
def main():
    print("Testing print_success_message from araras.kernel.monitoring")
    try:
        res = print_success_message("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
