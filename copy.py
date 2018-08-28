import os, shutil


def main(dir: str, suffix: str = None, aimSuffix: str = None):
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
            if file.endswith(suffix):
                # print('yes')
                print(os.path.join(i, file))
                copy(os.path.join(i, file), aimSuffix)
                # shutil.copyfile(os.path.join(i, file), os.path.join(i, file + aimSuffix))

                # print('======================')
                #     for f in k:
                #         files.append(f)
                # print(list(set(files)))


def copy(filename, aimSuffix):
    str = 'copy ' + filename + " " + filename + aimSuffix
    mystr = os.popen(str)  # popen与system可以执行指令,popen可以接受返回对象
    mystr = mystr.read()


if __name__ == "__main__":
    main(r'E:\workspace\ChatBot', suffix='java',
         aimSuffix='_')
