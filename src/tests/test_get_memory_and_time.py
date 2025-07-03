from araras.keras.utils.profiler import get_memory_and_time
from tests.helpers import make_model


def main():
    print("Testing get_memory_and_time from araras.keras.utils.profiler")
    model = make_model()
    try:
        res = get_memory_and_time(model, 1, "CPU:0", 1, 1, False)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
