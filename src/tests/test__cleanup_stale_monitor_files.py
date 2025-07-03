from araras.kernel.monitoring import _cleanup_stale_monitor_files
def main():
    print("Testing _cleanup_stale_monitor_files from araras.kernel.monitoring")
    try:
        res = _cleanup_stale_monitor_files()
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
