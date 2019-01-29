import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

N_SAMPLES = 30000

CHUNKS = 1024   

SD_SIG_SIZE = 60000
# DFT_S = 5000


class Sound:
    def __init__(self, name):
        self.spf = wave.open(name + '.wav', 'r')
        # Extract Raw Audio from Wav File
        self.signal = self.spf.readframes(-1)
        self.signal = np.fromstring(self.signal, 'Int16')
        self.fs = self.spf.getframerate()
        self.name = name
        # self.sample_from_signal()

    def save_plot(self):
        d = self.calc_fft()
        plt.title("DFT of input")
        plt.plot(d, color ="red")
        plt.savefig("static/up.png")

    def sample_from_signal(self):
        t = []
        step = int(self.signal.size / N_SAMPLES)
        for i in range(0, self.signal.size, step):
            t.append(self.signal[i])

        self.signal = np.array(t[:N_SAMPLES])
        

    def calc_fft(self):
        return np.fft.rfft(self.signal)

    def normalize(self):
        temp = []
        fdt = self.calc_fft()
        # print fdt.size
        
        step = int(fdt.size / N_SAMPLES)
        for i in xrange(0, fdt.size, step):
            temp.append(fdt[i])

        return np.array(temp[:N_SAMPLES])

    def normalize_squared(self):
        temp = []
        fdt = self.calc_fft()
        # print fdt.size
        
        step = int(fdt.size / N_SAMPLES)
        # step = int(self.signal.size / N_SAMPLES)
        # step = 1
        # print step
        for i in xrange(0, fdt.size, step):
            temp.append(fdt[i])

        # temp = np.array(temp[:N_SAMPLES])
        # print temp.size

        return np.square((temp[:N_SAMPLES]))

    def standard_signal(self):
        step = int(self.signal.size / SD_SIG_SIZE)
        print step
        res = []

        for i in range(1, self.signal.size + 1, step):
            res.append(i)
        
        return np.array(res)


    def calc_norm_regarding_to_time(self):
        signal = self.signal
        size_sig = signal.size

        amount_poss = int(size_sig / CHUNKS)
        # print size_sig

        result = [] # 2D
        for time in xrange(amount_poss):
            temp = [] # Chuks = len
            for i in xrange(CHUNKS):
                temp.append(signal[time*CHUNKS + i] + 0j)
            

            result.append(np.fft.fft(temp))
            # print np.fft.fft(temp).size
        
        return np.array(result)


    def show(self):

        # if self.spf.getnchannels() == 2:
        #     print 'Just mono files'
        #     sys.exit(0)

        # time_axis = np.linspace(0, len(self.signal)/self.fs, num=len(self.signal))

        fdt = self.calc_fft()
        plt.figure(2)
        plt.title('original signal')
        plt.plot(self.signal)
        plt.plot(fdt)
        plt.show()
        # plt.imsave(self.name + ".png", self.signal)
        # plt.savefig(self.name + "1.pdf")

    def show_spec_plot(self, arr):
        plt.figure(1)
        plt.title("plot")
        print arr.size
        plt.plot(arr)
        plt.show()


def calc_new_mse(input_signal):
    list_bale = []

    for i in range(1, 48):
        s = Sound('data/bale/%s' %i)
        list_bale.append(s.calc_norm_regarding_to_time())

    list_kheyr = []
    for i in range(1, 48):
        s = Sound('data/kheyr/%s' %i)
        list_kheyr.append(s.calc_norm_regarding_to_time())

    in_s = Sound(input_signal)
    t = in_s.calc_norm_regarding_to_time()

    min_bale = np.inf
    for i in list_bale:
        min_t = min(i.shape[0], t.shape[0])
        temp_2 = 0
        for k in range(min_t):
            temp = i[k] - t[k]
            temp = np.absolute(temp)
            temp = np.square(temp)
            temp = np.sum(temp)
            temp_2 += temp
        
        temp_2 = np.average(temp_2)
        if temp_2 < min_bale:
            min_bale = temp_2


    min_kheyr = np.inf
    for i in list_kheyr:
        min_t = min(i.shape[0], t.shape[0])
        temp_2 = 0
        for k in range(min_t):
            temp = i[k] - t[k]
            temp = np.absolute(temp)
            temp = np.square(temp)
            temp = np.sum(temp)
            temp_2 += temp
        
        temp_2 = np.average(temp_2)
        if temp_2 < min_kheyr:
            min_kheyr = temp_2


    if min_bale > min_kheyr:
        print "kheyr"
        return 0
    else:
        print "bale"
        return 1

    # print "bale: ", min_bale
    # print "kheyr:", min_kheyr
    

cache_bale = False
cache_kheyr = False
def calc_mse(input_signal):
    global cache_bale, cache_kheyr
    list_bale = []
    if cache_bale == False:
        for i in range(1, 48):
            s = Sound('data/bale/%s' % i)
            # s = Sound('data/train/yes/%s' % i)
            # t = s.normalize()
            t = s.normalize_squared()
            list_bale.append(t)
        cache_bale = list_bale
    
    list_bale = cache_bale

    if cache_kheyr == False:
        list_kheyr = []
        for i in range(1, 48):
            s = Sound('data/kheyr/%s' % i)
            # s = Sound('data/train/no/%s' % i)
            # t = s.normalize()
            t = s.normalize_squared()
            list_kheyr.append(t)
        cache_kheyr = list_kheyr
    list_kheyr = cache_kheyr

    in_s = Sound(input_signal)
    # t = in_s.normalize()
    t = in_s.normalize_squared()

    # plt.plot(t)
    # plt.plot(list_bale[3], color="g")
    # plt.plot(list_kheyr[5], color='r')
    # plt.show()

    min_bale = np.inf
    for i in list_bale:
        temp = np.subtract(i, t)
        temp = np.absolute(temp)
        temp = np.square(temp)
        temp = np.sum(temp)
        if temp < min_bale:
            min_bale = temp
    
    min_kheyr = np.inf
    for i in list_kheyr:
        temp = np.subtract(i, t)
        temp = np.absolute(temp)
        temp = np.square(temp)
        temp = np.sum(temp)
        if temp < min_kheyr:
            min_kheyr = temp
    
    if min_bale > min_kheyr:
        print "kheyr"

        return 0, min_bale, min_kheyr
    else:
        print "bale"

        return 1, min_bale, min_kheyr

    print "bale: ", min_bale
    print "kheyr:", min_kheyr


def calc_by_avg_mse(input_signal):
    list_bale = []
    for i in range(1, 48):
        # s = Sound('data/bale/%s' % i)
        s = Sound('data/bale/%s' % i)
        # t = s.normalize()
        t = s.normalize_squared()
        list_bale.append(t)

    avg_bale = np.average(list_bale)
    
    list_kheyr = []
    for i in range(1, 48):
        # s = Sound('data/kheyr/%s' % i)
        s = Sound('data/kheyr/%s' % i)
        # t = s.normalize()
        t = s.normalize_squared()
        list_kheyr.append(t)
    
    avg_kheyr = np.average(list_kheyr)

    in_s = Sound(input_signal)
    # t = in_s.normalize()
    t = in_s.normalize_squared()

    bale_possiblity = np.subtract(t, avg_bale)
    bale_possiblity = np.absolute(bale_possiblity)
    bale_possiblity = np.square(bale_possiblity)
    bale_possiblity = np.sum(bale_possiblity)

    kheyr_possiblity = np.subtract(t, avg_kheyr)
    kheyr_possiblity = np.absolute(kheyr_possiblity)
    kheyr_possiblity = np.square(kheyr_possiblity)
    kheyr_possiblity = np.sum(kheyr_possiblity)

    if bale_possiblity < kheyr_possiblity:
        print "bale"
        return 1
    else:
        print "kheyr"
        return 0

    print bale_possiblity
    print kheyr_possiblity


def tensor(input_signal):
    import tensorflow as tf
    from tensorflow import keras

    list_bale = []
    for i in range(1, 48):
        # s = Sound('data/bale/%s' % i)
        s = Sound('data/bale/%s' % i)
        # t = s.normalize()
        t = s.normalize_squared()
        list_bale.append(t)

    list_bale = np.array(list_bale)
    
    list_kheyr = []
    for i in range(1, 48):
        # s = Sound('data/kheyr/%s' % i)
        s = Sound('data/kheyr/%s' % i)
        # t = s.normalize()
        t = s.normalize_squared()
        list_kheyr.append(t)

    train_data = np.array([list_kheyr, list_bale])
    train_label = np.array([0, 1])

    model = keras.Sequential(
        [keras.layers.Dense(30000, activation=tf.nn.relu),
        keras.layers.Dense(30000, activation=tf.nn.softmax)]
    )

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(train_data, train_label, epochs=1)
    


def calc_with_data_test(func_test):
    bale_old = 0
    for i in range(1, 13):
        print "bale", 'data: ', i
        # r = func_test('data/test_b/%s' %i)
        r = func_test('data/test_b/%s' %i)
        if r[0] == 1:
            bale_old += 1
    kheyr_old = 0
    for i in range(1, 13):
        print "kheyr", 'data: ', i
        # r = func_test('data/test_kh/%s' %i)
        r = func_test('data/test_kh/%s' %i)
        if r[0] == 0:
            kheyr_old += 1


    print "perc b, kh: ", bale_old/12.0, kheyr_old/12.0

# import random
# i, j = random.randint(1, 48), random.randint(1, 48)
# s1 = Sound('data/bale/%s' %i)
# s2 = Sound('data/kheyr/%s' %j)

# plt.figure(2)

# plt.plot(s1.calc_fft())
# plt.plot(s2.calc_fft(), color="black")
# plt.show()

# calc_with_data_test(calc_mse)
# s = Sound('data/bale/1')
# s.save_plot()