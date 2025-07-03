from araras.kernel.monitoring import start_monitor
def main():
    print("Testing start_monitor from araras.kernel.monitoring")
    try:
        res = start_monitor("dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
