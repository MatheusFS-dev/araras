from araras.utils.gpu import get_user_gpu_choice
def main():
    print("Testing get_user_gpu_choice from araras.utils.gpu")
    try:
        res = get_user_gpu_choice()
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
