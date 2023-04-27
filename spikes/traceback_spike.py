import traceback


def func1():
    func2()


def func2():
    func3()


def func3():
    try:
        raise Exception('This is the error message.')
    except Exception as e:
        print('\n'.join(traceback.TracebackException.from_exception(e).format()))


func1()
