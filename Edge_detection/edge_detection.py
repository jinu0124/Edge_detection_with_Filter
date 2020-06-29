import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import math
import random

# agent.py 에서 실행(Ctrl+Shift+F10) 부탁드립니다.
class edge:
    # 추후 CUDA 프로그래밍(PyCUDA) 혹은 Tensorflow를 사용하여 GPU 연산으로 Conv(회선)을 더욱 빠르게 수행 가능
    def __init__(self, ROOT_DIR):
        super().__init__()
        self.ROOT_DIR = ROOT_DIR
        print('2015253039 권진우 영상처리 과제 HW3')

        print("**********출력(plt.show())은 주석처리 되어있고 결과 이미지가 저장만 되도록 설정되어있습니다.**********\n**********출력하여 확인을 위해서는 # plt.show()주석을 빼주십시오.**********")
        # Image Import
        lena_image = self.import_image(ROOT_DIR)

        # make masks
        mask_dict = self.make_mask()

        while(1):
            # User Control Method
            self.control(mask_dict, lena_image)

    # 전체적인 User Control & PyQt 등의 패키지를 활용하여 GUI 프로그램으로 만들 시 Control 내에 Control 관련 함수 구현으로 쉽게 개발 가능
    def control(self, mask_dict, lena_image):
        print('1. 기본 lena image에 Mask OP')
        print('2. 가우시안 노이즈 lena image에 Mask OP')
        print('3. 가우시안 노이즈 lena image 보기')
        print('4. 각 Mask 별 Edge Detection err Rate 확인')
        print('5. 문제 1번 Edge Detection 모두 수행하기(Noise+일반 * 4 Masks)')
        print('6. Boat image에 노이즈 & 필터별 MSE 확인')
        print('0. 종료하기')
        get = input('숫자 번호 입력')
        try:
            get = int(get)
        except:
            print('숫자를 입력해주세요.')
            return

        if get is 0:
            exit()
        elif get is 1:
            print('1. roberts mask')
            print('2. sobel mask')
            print('3. prewitt mask')
            print('4. stochastic mask')
            print('input any number to Undo')
            get2 = input('숫자 번호 입력')
            try:
                get2 = int(get2)
            except:
                print('숫자를 입력해주세요.')
                return

            # apply masks, Show & Store Image(Original Image's Edge Detection Result), flag means origin/noise flag
            if get2 is 1:
                self.apply_roberts(mask_dict['rbt_r'], mask_dict['rbt_c'], lena_image, 0)
            elif get2 is 2:
                self.apply_sobel(mask_dict['sbl_r'], mask_dict['sbl_c'], lena_image, 0)
            elif get2 is 3:
                self.apply_prewitt(mask_dict['prt_r'], mask_dict['prt_c'], lena_image, 0)
            elif get2 is 4:
                self.apply_stochastic(mask_dict['sto_r'], mask_dict['sto_c'], lena_image, 0)
            else:
                return

        elif get is 2 or get is 3:
            print('making noise image...')
            # get variance from Image
            stddev_noise = self.variance(lena_image, 8.0)

            # get Gaussian Noise Lena Image & Compute
            noise_image = self.make_noise_image(lena_image, stddev_noise)
            if get is 2:
                print('1. roberts mask with noise')
                print('2. sobel mask with noise')
                print('3. prewitt mask with noise')
                print('4. stochastic mask with noise')
                flag = input('숫자 번호 입력')
                try:
                    flag = int(flag)
                except:
                    print('숫자를 입력해주세요.')
                    return
                if flag is 1:
                    self.apply_roberts(mask_dict['rbt_r'], mask_dict['rbt_c'], noise_image, 1)
                elif flag is 2:
                    self.apply_sobel(mask_dict['sbl_r'], mask_dict['sbl_c'], noise_image, 1)
                elif flag is 3:
                    self.apply_prewitt(mask_dict['prt_r'], mask_dict['prt_c'], noise_image, 1)
                elif flag is 4:
                    self.apply_stochastic(mask_dict['sto_r'], mask_dict['sto_c'], noise_image, 1)
                else:
                    return
            if get is 3:
                plt.imshow(noise_image, cmap='gray')
                plt.imsave('noise_image.bmp', noise_image, cmap='gray')
                plt.show()

        elif get is 4:
            self.compute_err_rate(mask_dict)
        elif get is 5:
            self.compute(mask_dict)
        elif get is 6:
            self.boat_raw()

        return

    # Output Image 에 대한 저장 기능을 모듈화하여 간단하게 폴더를 만들어서 내부에 저장하도록 구현
    def save_image(self, dst_image, flag, name):
        if flag is 1:
            if os.path.exists('Noise_Lena_Edge_Detecting'): # 해당 폴더가 존재한다면
                plt.imsave('Noise_Lena_Edge_Detecting/' + name + '.bmp', dst_image, cmap='gray')
            else:
                os.makedirs('Noise_Lena_Edge_Detecting')
                plt.imsave('Noise_Lena_Edge_Detecting/' + name + '.bmp', dst_image, cmap='gray')
        elif flag is 0:
            if os.path.exists('Original_Lena_Edge_Detecting'):
                plt.imsave('Original_Lena_Edge_Detecting/' + name + '.bmp', dst_image, cmap='gray')
            else:
                os.makedirs('Original_Lena_Edge_Detecting')
                plt.imsave('Original_Lena_Edge_Detecting/' + name + '.bmp', dst_image, cmap='gray')

    def apply_roberts(self, roberts_row, roberts_col, lena_image, flag):
        roberts_r = roberts_row
        roberts_c = roberts_col

        dst_image = self.mask_operator(roberts_r, roberts_c, lena_image) # Gx, Gy, image

        plt.imshow(dst_image, cmap='gray') # cmap = color map
        # plt.show()
        self.save_image(dst_image, flag, name='roberts')

    def apply_sobel(self, sobel_row, sobel_col, lena_image, flag):
        sobel_r = sobel_row
        sobel_c = sobel_col

        # if not os.path.exists('sobel.jpg'):
        dst_image = self.mask_operator(sobel_r, sobel_c, lena_image)

        plt.imshow(dst_image, cmap='gray')
        # plt.show()
        self.save_image(dst_image, flag, name='sobel')

    def apply_prewitt(self, prewitt_row, prewitt_col, lena_image, flag):
        prewitt_r = prewitt_row
        prewitt_c = prewitt_col

        dst_image = self.mask_operator(prewitt_r, prewitt_c, lena_image)

        plt.imshow(dst_image, cmap='gray')
        # plt.show()
        self.save_image(dst_image, flag, name='prewitt')

    def apply_stochastic(self, stoc_r, stoc_c, lena_image, flag):
        dst_image = self.mask_operator(stoc_r, stoc_c, lena_image)

        plt.imshow(dst_image, cmap='gray')
        # plt.show()
        self.save_image(dst_image, flag, name='stochastic')

    # mask Operator : Gx, Gy, Combine, Padding + return dst_image
    def mask_operator(self, mask_r, mask_c, lena_image): # Gx, Gy, Original image
        dst_image = [[0 for i in range(512)] for j in range(512)] # 512x512 배열 '0' 초기화
        dst_r_image = np.zeros((512,512), dtype=int)
        dst_c_image = np.zeros((512,512), dtype=int)

        a = int(len(mask_r) / 2)

        print("Gx Mask Start")
        # X 방향 -> Operate Mask Compute
        for r in range(a, len(lena_image[0]) - round(len(mask_r) / 2)):
            for c in range(a, len(lena_image) - round(len(mask_c) / 2)):  # 계산될 Pixel 1개에 대해서 수행
                new_value = 0
                # 각 Pixel에 대해서 3x3 sobel mask를 적용하여 Gradient Compute
                for i in range(len(mask_r)):
                    for j, b in enumerate(mask_r[i]):  # b는 mask의 가중치가 된다.
                        if b != 0:
                            # 3x3 mask이므로 i, j for문이 3번씩 돈다
                            new_value += b * lena_image[r + i - a][c + j - a]
                dst_r_image[r][c] = new_value
        print('Gx Done')

        print('Gy Mask Start')
        # Y 방향 | Operate Mask Compute
        for r in range(a, len(lena_image[0]) - round(len(mask_r) / 2)):
            for c in range(a, len(lena_image) - round(len(mask_c) / 2)):
                new_value = 0
                for i, e in enumerate(mask_c):
                    for j, b in enumerate(mask_c[i]):
                        if b != 0:
                            new_value += b * lena_image[r + i - a][c + j - a]
                dst_c_image[r][c] = new_value
        print('Gy Done')

        # Mask Compute가 수행되지 않은 가장자리 Pixels에 대해서 원래의 이미지 값 적용
        for i in range(len(dst_image[0])):
            dst_image[0][i] = lena_image[0][i]
            dst_image[i][0] = lena_image[i][0]
            dst_image[511][i] = lena_image[511][i]
            dst_image[i][511] = lena_image[i][511]

        # Gx, Gy Combine and Threshold : 150
        for r in range(512):
            for c in range(512):
                # val = abs(dst_r_image[r][c]) + abs(dst_c_image[r][c])
                val = ((dst_r_image[r][c] ** 2) + (dst_c_image[r][c] ** 2)) ** 0.5
                if val > 150:
                    dst_image[r][c] = 255
                else:
                    dst_image[r][c] = 0
                # dst_image[r][c] = int(val)

        return dst_image

    # mask에 대한 행렬 정보를 담고 있는 make_mask함수
    def make_mask(self):
        masks = dict()

        roberts_m_r = ([0, 0, -1],
                    [0, 1, 0],
                    [0, 0, 0])
        roberts_m_c = ([1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 0])

        sobel_m_r = ([1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1])
        sobel_m_c = ([-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1])

        prewitt_m_r = ([1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1])
        prewitt_m_c = ([-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1])

        stochastic_m_r = ([0.267, 0.364, 0, -0.364, -0.267],
                          [0.373, 0.562, 0, -0.562, -0.373],
                          [0.463, 1.000, 0, -1.000, -0.463],
                          [0.373, 0.562, 0, -0.562, -0.373],
                          [0.267, 0.364, 0, -0.364, -0.267])
        stochastic_m_c = ([-0.267, -0.373, -0.463, -0.373, -0.267],
                          [-0.364, -0.562, -1.000, -0.562, -0.364],
                          [0, 0, 0, 0, 0],
                          [0.364, 0.562, 1.000, 0.562, 0.364],
                          [0.267, 0.373, 0.463, 0.373, 0.267])

        masks = {'rbt_r': roberts_m_r, 'rbt_c': roberts_m_c, 'sbl_r': sobel_m_r, 'sbl_c':sobel_m_c, 'prt_r': prewitt_m_r, 'prt_c': prewitt_m_c,
                 'sto_r': stochastic_m_r, 'sto_c': stochastic_m_c}

        return masks

    def import_image(self, ROOT_DIR):
        intensity_list = []
        lena_image = np.zeros((512, 512), dtype=int)

        self.image_name = 'lena_bmp_512x512_new.bmp'

        if os.path.isfile(os.path.join(ROOT_DIR, self.image_name)):
            lena = Image.open(os.path.join(ROOT_DIR, self.image_name))
            lena_image = np.array(lena)
            self.lena_image = lena_image

        # for r in lena_image:
        #     for c in lena_image[r]:
        #         intensity_list.append(c)

        return lena_image

    # Error Rate 구하기
    def compute_err_rate(self, mask):
        if os.path.exists('Noise_Lena_Edge_Detecting') and os.path.exists('Original_lena_Edge_Detecting'):
            try:
                noise_dir = os.path.join(self.ROOT_DIR, 'Noise_Lena_Edge_Detecting')
                origin_dir = os.path.join(self.ROOT_DIR, 'Original_Lena_Edge_Detecting')
                noise_roberts = np.array(Image.open(os.path.join(noise_dir, 'roberts.bmp')).convert('L')) # Read시 Gray Scale로 읽기
                noise_sobel = np.array(Image.open(os.path.join(noise_dir, 'sobel.bmp')).convert('L'))
                noise_prewitt = np.array(Image.open(os.path.join(noise_dir, 'prewitt.bmp')).convert('L'))
                noise_stochastic = np.array(Image.open(os.path.join(noise_dir, 'stochastic.bmp')).convert('L'))
                origin_roberts = np.array(Image.open(os.path.join(origin_dir, 'roberts.bmp')).convert('L'))
                origin_sobel = np.array(Image.open(os.path.join(origin_dir, 'sobel.bmp')).convert('L'))
                origin_prewitt = np.array(Image.open(os.path.join(origin_dir, 'prewitt.bmp')).convert('L'))
                origin_stochastic = np.array(Image.open(os.path.join(origin_dir, 'stochastic.bmp')).convert('L'))

                # 기존의 엣지 Detection에서 새로운 edge나 사라진 edge의 개수
                rbt1 = np.count_nonzero(origin_roberts - noise_roberts)  # n1
                sbl1 = np.count_nonzero(origin_sobel - noise_sobel) # origin에서 noise를 빼면 origin에만 있던 점은 255가 남고, noise에만 있던 edge는 'uint8'type이므로 0 - 255 => 1이 남는다.
                prt1 = np.count_nonzero(origin_prewitt - noise_prewitt)
                sto1 = np.count_nonzero(origin_stochastic - noise_stochastic)

                # 기존의 Edge의 개수
                rbt0 = np.count_nonzero(origin_roberts)  # n0
                sbl0 = np.count_nonzero(origin_sobel)
                prt0 = np.count_nonzero(origin_prewitt)
                sto0 = np.count_nonzero(origin_stochastic)

                print('에러율')
                print('Roberts :', rbt1 / rbt0)
                print('sobel :', sbl1 / sbl0)
                print('prewitt :', prt1 / prt0)
                print('stochastic :', sto1 / sto0)
            except:
                print('에러율 계산을 위한 이미지가 준비되지 않았습니다. \n이미지를 준비하기위해 Masking작업을 시작하시겠습니까?(오래걸림 주의) \n1. 예  2. 아니오 \n')
                a = int(input())
                if a == 2:
                    return
                else:
                    self.compute(mask)
        else:
            print('에러율 계산을 위한 이미지가 준비되지 않았습니다. \n이미지를 준비하기위해 Masking작업을 시작하시겠습니까?(오래걸림 주의) \n1. 예  2. 아니오 \n')
            a = int(input())
            if a == 2:
                return
            self.compute(mask)
            return

    # Noise영상에 대한 Edge Detection & 기존 영상에 대한 Edge Detection 모두 한번에 수행 함수
    def compute(self, mask):
        print('Original image Masking...')
        self.apply_roberts(mask['rbt_r'], mask['rbt_c'], self.lena_image, flag=0)
        self.apply_sobel(mask['sbl_r'], mask['sbl_c'], self.lena_image, flag=0)
        self.apply_prewitt(mask['prt_r'], mask['prt_c'], self.lena_image, flag=0)
        self.apply_stochastic(mask['sto_r'], mask['sto_c'], self.lena_image, flag=0)

        print('Noise image Masking...')
        # get Gaussian Noise Lena Image & Compute # variance 함수 구현을 통해 variance 값 return
        stddev_noise = self.variance(self.lena_image, 8.0)
        noise_image = self.make_noise_image(self.lena_image, stddev_noise)

        self.apply_roberts(mask['rbt_r'], mask['rbt_c'], noise_image, flag=1)
        self.apply_sobel(mask['sbl_r'], mask['sbl_c'], noise_image, flag=1)
        self.apply_prewitt(mask['prt_r'], mask['prt_c'], noise_image, flag=1)
        self.apply_stochastic(mask['sto_r'], mask['sto_c'], noise_image, flag=1)
        print('Masking Done & If you want to see Error Rate, input \'4\'')

    # variance 구하기
    def variance(self, lena_image, n=8.0):
        MN = len(lena_image)*len(lena_image[0]) # 262144
        sum_x = 0.0 # x
        sum_x_2 = 0.0 # x^2

        for i in range(len(lena_image)):
            for c in lena_image[i]:
                sum_x_2 += c**2
                sum_x += c

        sigma_2 = (sum_x_2/float(MN)) - ((sum_x/float(MN))**2) # variance 공식

        stddev_noise = float(math.sqrt(sigma_2/pow(10.0, n/10))) # SNR : 8dB Noise
        return stddev_noise

    # making Noise image from input Image & return Noise Image
    def make_noise_image(self, lena_image, stddev_noise):
        # noise image를 담을 512x512 배열
        noise_image = np.zeros((len(lena_image[0]), len(lena_image)), dtype=int)
        sigma = stddev_noise

        # Gaussian Compute
        def gaussian(sd):
            ready = 0
            gstore, v1, v2, r, fac, gaus = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            r1, r2 = 0, 0

            if ready is 0:
                a = 0
                while (r > 1.0 or a is 0):
                    a = 1
                    r1 = random.randrange(0, 32767) # C에서 RAND_MAX는 32767이다.(2^16)
                    r2 = random.randrange(0, 32767)
                    v1 = 2 * (float(r1) / float(32767) - 0.5)
                    v2 = 2 * (float(r2) / float(32767) - 0.5)
                    r = v1 ** 2 + v2 ** 2

                fac = float(math.sqrt(float(-2 * math.log(r) / r)))

                gstore = v1 * fac
                gaus = v2 * fac
                ready = 1
            else:
                ready = 0
                gaus = gstore

            return gaus * sd

        for i in range(len(lena_image)):
            for j, e in enumerate(lena_image[i]):
                s = e + gaussian(sigma) # 픽셀에 gaussian noise를 입힌다.
                a = 255 if s > 255 else s # 255보다 크면 255로 클램핑
                a = 0 if a < 0 else a # 0보다 작으면 0으로 클램핑
                noise_image[i][j] = a

        return noise_image

    # BOAT512에 대해 Noise & MSE Check about lowpass, median Filter
    def boat_raw(self):
        self.boat_image = np.zeros((512,512), dtype=int)

        # BOAT512.raw파일을 2차원 Numpy Array 형식으로 import
        def get_boat_raw():
            boat_image_dir = os.path.join(self.ROOT_DIR, 'BOAT512.raw')
            scene_infile = open(boat_image_dir, 'rb')
            scene_image_array = np.fromfile(scene_infile, dtype=np.uint8, count=512 * 512)
            boat_image = Image.frombuffer("L", [512, 512], scene_image_array, 'raw', 'L', 0, 1) # L : 8bit gray Scale
            boat_image = np.array(boat_image) # numpy 2차 배열에 담기
            plt.imsave('boat_image.bmp', boat_image, cmap='gray')

            return boat_image

        # get BOAT512 two-Dimension Array
        boat_image = get_boat_raw()
        self.boat_image = boat_image

        # make SNR=9DB Noise Image for BOAT512
        stddev_noise = self.variance(boat_image, n=9.0) # SNR = 9.0으로 계산
        noise_image = self.make_noise_image(boat_image, stddev_noise) # Lena영상의 Gaussian Noise 만들기와 동일

        plt.imshow(noise_image, cmap='gray')
        # plt.show()
        plt.imsave('boat_noise.bmp', noise_image, cmap='gray')

        print('Median Filtering Start...')
        # OP Median Filter(3x3)
        self.median(noise_image)

        print('Lowpass Filtering Start...')
        # OP Lowpass Filter(3x3)
        self.lowpass(noise_image)

    # filtering : lowpass and median # flag에 따라서 median인지 lowpass인지 분별하여 해당 mask operating
    # lowpass와 median 필터는 별도로 mask를 만들어둘 필요가 없다. Ex. lowpass의 경우 for문 내에서 1/9씩 곱셈합을 구하면 되기 때문
    def filtering(self, image, flag='0'):
        mask_size = 3
        a = int(mask_size / 2)
        window_list = []
        new_image = [[0 for i in range(512)] for j in range(512)]  # == np.zeros((512,512), dtype=int)

        # quicksort 구현함수
        def quicksort(window_list, left, right):
            if left >= right:
                return
            window_list[int((left + right) / 2)], window_list[left] = window_list[left], window_list[int((left + right) / 2)]
            last = left
            for i in range(left + 1, right + 1):
                if window_list[i] < window_list[left]:
                    window_list[++last], window_list[i] = window_list[i], window_list[++last] # ++last : last+1 배열로 접근 + last값 +1 증가
            window_list[last], window_list[left] = window_list[left], window_list[last]
            quicksort(window_list, left, last - 1)
            quicksort(window_list, last + 1, right)

        # list에서 중간 값 찾기 함수
        def get_mid(window_list):
            left, right = 0, 8
            quicksort(window_list, left, right) # quicksort Call
            return window_list[4]

        # median filtering
        if flag is 'median':
            for r in range(len(image)-a-1): # 0,0부터 시작하되 3x3 Mask 적용이기에 0~509번 Pixel까지 진행
                for c in range(len(image[r])-a-1):
                    for w in range(3): # 3x3 filter이므로 3x3 for문
                        for k in range(3):
                            val = image[r+w][c+k]
                            window_list.append(val)
                    # new_image[r+1][c+1] = statistics.median(window_list)
                    new_image[r+1][c+1] = get_mid(window_list) # 좌상단(0,0)을 중심으로 mask를 시작하였으므로 (r+1, c+1)좌표에 적용
                    window_list = [] # window_list초기화(1개의 픽셀에 대하여 get_mid 수행 후 초기화 필요)
        # lowpass filtering
        elif flag is 'lowpass':
            val = 0
            for r in range(len(image)-a-1):
                for c in range(len(image[r])-a-1):
                    for w in range(3):
                        for k in range(3):
                            val += 1/9*image[r+w][c+k] # 1/9를 가중치로 곱하여 누적합을 구함으로써 Lowpass Filtering
                    new_image[r+1][c+1] = val
                    val = 0

        # Mask Compute가 수행되지 않은 가장자리 Pixels에 대해서 원래의 이미지 값 적용
        for i in range(len(new_image[0])):
            new_image[0][i] = image[0][i]
            new_image[i][0] = image[i][0]
            new_image[511][i] = image[511][i]
            new_image[i][511] = image[i][511]

        return new_image

    def lowpass(self, image):
        new_image = self.filtering(image, flag='lowpass')
        plt.imshow(new_image, cmap='gray')
        # plt.show()
        plt.imsave('boat_lowpass.bmp', new_image, cmap='gray')

        # Mean Square Error
        mse = self.MSE(new_image)
        print('Mean Square Error in lowpass :', mse)

    def median(self, image):
        new_image = self.filtering(image, flag='median')
        plt.imshow(new_image, cmap='gray')
        # plt.show()
        plt.imsave('boat_median.bmp', new_image, cmap='gray')

        # Mean Square Error
        mse = self.MSE(new_image)
        print('Mean Square Error in median :', mse)

    # MEAN Square Error를 계산
    def MSE(self, image):
        Accum = 0.0
        for r in range(len(image)):
            for c in range(len(image[r])):
                Accum += (image[r][c] - self.boat_image[r][c])**2 # 차이값^2을 누적하여서
        mse = Accum/(len(image)*len(image[0])) # 전체 픽셀 수로 나누기
        return mse
