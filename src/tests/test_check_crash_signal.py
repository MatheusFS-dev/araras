from araras.kernel.monitoring import check_crash_signal
def main():
    print("Testing check_crash_signal from araras.kernel.monitoring")
    try:
        res = check_crash_signal("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
