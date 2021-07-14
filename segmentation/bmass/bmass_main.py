# -*- coding: utf-8 -*-

import os
import matlab.engine


def bmass_infer(data_path, output_path):
    odds = 1.667
    with open('params.txt', 'w') as f:
        f.write(data_path + '\n')
        f.write(output_path + '\n')
        f.write(str(odds) + '\n')
        f.close()

    engine = matlab.engine.start_matlab()
    engine.bmass_main()
    engine.exit()


if __name__ == '__main__':
    data_path = '../../data/'
    output_path = '../../output/seg/'
    os.makedirs(output_path)

    bmass_infer(data_path, output_path)