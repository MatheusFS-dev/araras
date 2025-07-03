from araras.kernel.monitoring import print_restart_info
def main():
    print("Testing print_restart_info from araras.kernel.monitoring")
    try:
        res = print_restart_info("dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
