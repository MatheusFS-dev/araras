from araras.utils.logs import log_resources
def main():
    print("Testing log_resources from araras.utils.logs")
    try:
        res = log_resources("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
