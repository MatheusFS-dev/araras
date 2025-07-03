from araras.kernel.monitoring import stop_monitor
def main():
    print("Testing stop_monitor from araras.kernel.monitoring")
    try:
        res = stop_monitor("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
