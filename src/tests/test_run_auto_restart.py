from araras.kernel.monitoring import run_auto_restart
def main():
    print("Testing run_auto_restart from araras.kernel.monitoring")
    try:
        res = run_auto_restart("dummy", "dummy", "dummy", "dummy", "dummy", "dummy", "dummy", "dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
