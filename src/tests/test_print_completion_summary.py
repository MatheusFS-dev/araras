from araras.kernel.monitoring import print_completion_summary
def main():
    print("Testing print_completion_summary from araras.kernel.monitoring")
    try:
        res = print_completion_summary("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
