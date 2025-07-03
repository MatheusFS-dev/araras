from araras.utils.dir import create_run_directory
def main():
    print("Testing create_run_directory from araras.utils.dir")
    try:
        res = create_run_directory("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
