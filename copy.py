import os, shutil

aimSuffix = '_'
suffixs = ['java', 'text']
suffixs2 = ['java_', 'text_']

path = r'E:\test\ChatBot'


def main(dir: str, mode):
    """
    :param dir:主目录
    :param suffix:要找的文件的后缀
    :param aimSuffix: 要加的后缀
    :return:
    """
    files = []
    for i, j, k in os.walk(dir):
        # print(i)
        # print(j)
        # print(k)
        for file in k:
            # print(file)
            # print(file.endswith('xml'))
            if (mode == 0):
                if (file.endswith('java')) or file.endswith('txt'):
                    # print('yes')
                    print(os.path.join(i, file))
                    copy(os.path.join(i, file))
                    # shutil.copyfile(os.path.join(i, file), os.path.join(i, file + aimSuffix))
            elif (mode == 1):
                if (file.endswith('java_')) or file.endswith('txt_'):
                    restore(os.path.join(i, file))
            else:
                if (file.endswith('java')) or file.endswith('txt'):
                    os.remove(os.path.join(i, file))


def copy(filename):
    str = 'copy ' + filename + " " + filename + aimSuffix
    mystr = os.popen(str)  # popen与system可以执行指令,popen可以接受返回对象
    mystr = mystr.read()


def restore(filename):
    os.rename(filename, filename[:-1])


if __name__ == "__main__":
    main(path, mode=0)
    main(path, mode=2)
    main(path, mode=1)
