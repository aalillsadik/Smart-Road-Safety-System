from subprocess import call
import multiprocessing

def drowsiness():
    call(["python", "drowsiness.py"])

def talking():
    call(["python", "phone.py"])

drowsy = multiprocessing.Process(target=drowsiness)
talk = multiprocessing.Process(target=talking)

if __name__ == '__main__':
    talk.start()
    drowsy.start()
