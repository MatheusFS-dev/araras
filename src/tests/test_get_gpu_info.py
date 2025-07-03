from araras.utils.gpu import get_gpu_info
def main():
    print("Testing get_gpu_info from araras.utils.gpu")
    try:
        res = get_gpu_info()
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
