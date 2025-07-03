from araras.utils.gpu import gpu_summary
def main():
    print("Testing gpu_summary from araras.utils.gpu")
    try:
        res = gpu_summary()
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
