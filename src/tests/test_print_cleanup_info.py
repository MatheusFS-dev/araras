from araras.kernel.monitoring import print_cleanup_info
def main():
    print("Testing print_cleanup_info from araras.kernel.monitoring")
    try:
        res = print_cleanup_info("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
