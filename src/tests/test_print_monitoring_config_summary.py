from araras.kernel.monitoring import print_monitoring_config_summary
def main():
    print("Testing print_monitoring_config_summary from araras.kernel.monitoring")
    try:
        res = print_monitoring_config_summary("dummy", "dummy", "dummy", "dummy", "dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
