from araras.kernel.monitoring import print_process_status
def main():
    print("Testing print_process_status from araras.kernel.monitoring")
    try:
        res = print_process_status("dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
