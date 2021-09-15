from gf256 import GF256
import socket
import random
from itertools import combinations, combinations_with_replacement
import functools
import time


# lis(16)->lis(4 * 4)
def reshape_16(lis):
    res = [[] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            res[i].append(lis[4 * i + j])
    return res


def comb(n, b):
    res = 1
    for i in range(n - b + 1, n + 1):
        res *= i
    for i in range(1, b + 1):
        res //= i
    return res


def axis_1D(axis):
    return (axis//4, axis%4)


def power(x, n):
    mul = GF256(1)
    for _ in range(n):
        mul *= x
    return mul


def gen_comb_eff(powers):
    comb_dic = set()
    for i in powers:
        for j in range(i):
            if comb(i,j) % 2:
                comb_dic.add((j,i))
    return comb_dic


def gen_rand_gf256():
    return GF256(random.randint(0, 255))


class Player():
    Num_player = 0
    TTP = None

    def __init__(self,ip='localhost', rec_port=5000):
        self.no = Player.Num_player
        Player.Num_player += 1
        self.ip = ip
        self.rec_port = rec_port
        self.broadcast = None

    def set_ttp(self, ttp):
        Player.TTP = ttp

    def set_broadcast(self, value):
        self.broadcast = value

    def send_num(self,lis, target_ip, target_port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((target_ip, target_port))
        lis = [int(i) for i in lis]
        s.send(str(lis).encode())
        s.close()

    def prep_rec(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", self.rec_port))
        s.listen(1)
        self.conn, self.addr = s.accept()
        self.soc=s
        data = self.conn.recv(10000).decode()
        data = [GF256(i) for i in eval(data)]
        self.conn.close()
        self.rec_port += 1
        self.other = data

    def after_broadcast(self):
        return [self.broadcast[i]+self.other[i] for i in range(len(self.broadcast))]

    # this function is used by input_player and trusted_third_player
    # input-- inputs: list
    # output -- shares : two-dim list(first-dim:player, second-dim: additive secret sharing)
    def calculate_share(self, inputs):
        shares = [[] for _ in range(ComputePlayer.ComputeNum)]
        for i in inputs:
            sum = GF256(0)
            for j in range(ComputePlayer.ComputeNum - 1):
                temp = gen_rand_gf256()
                shares[j].append(temp)
                sum += temp
            shares[ComputePlayer.ComputeNum-1].append(i - sum)
        return shares

    def calculate_share_mac(self, inputs):
        shares = [[] for _ in range(ComputePlayer.ComputeNum)]
        for i in inputs:
            sum1 = GF256(0)
            sum2 = GF256(0)
            for j in range(ComputePlayer.ComputeNum - 1):
                temp1 = gen_rand_gf256()
                temp2 = gen_rand_gf256()
                shares[j].append((temp1, temp2))
                sum1 += temp1
                sum2 += temp2
            shares[ComputePlayer.ComputeNum-1].append((i[0] - sum1, i[1]-sum2))
        return shares


class InputPlayer(Player):
    def __init__(self, ip='localhost', rec_port=5000):
        super().__init__(ip, rec_port)

    def generate_keys(self, inputs=None, storage='memory'):
        if not inputs:
            inputs = [gen_rand_gf256() for i in range(16)]
        inputs = [(i, i*Player.TTP.mac_sum) for i in inputs]
        shares = self.calculate_share_mac(inputs)
        if storage == 'memory':
            for i in range(ComputePlayer.ComputeNum):
                ComputePlayer.ComputeList[i].set_keys(shares[i])
        elif storage == 'file':
            for i in range(ComputePlayer.ComputeNum):
                with open('key_P{}.txt'.format(i), 'w') as file:

                    file.write(str(len(shares[i])))
                    file.write('\n')
                    for j in shares[i]:
                        file.write(str(j))
                        file.write('\n')

    # encode plain_text
    # decode cipher_text
    def generate_texts(self, inputs=None, storage='memory'):
        if not inputs:
            inputs = [gen_rand_gf256() for i in range(16)]
        inputs = [(i, i * Player.TTP.mac_sum) for i in inputs]
        shares = self.calculate_share_mac(inputs)
        if storage == 'memory':
            for i in range(ComputePlayer.ComputeNum):
                ComputePlayer.ComputeList[i].set_texts(shares[i])
        elif storage == 'file':
            for i in range(ComputePlayer.ComputeNum):
                with open('text_P{}.txt'.format(i), 'w') as file:
                    print(len(shares[i]))
                    file.write(str(len(shares[i])))
                    file.write('\n')
                    for j in shares[i]:
                        file.write(str(j))
                        file.write('\n')


class ComputePlayer(Player):
    ComputeNum = 0
    ComputeList = []
    multiple_time = 0

    def __init__(self, ip='localhost', rec_port=5000):
        super().__init__(ip, rec_port)
        self.others = ComputePlayer.ComputeList[:]
        for p in self.others:
            p.others.append(self)
        ComputePlayer.ComputeList.append(self)
        self.compute_no = ComputePlayer.ComputeNum
        ComputePlayer.ComputeNum += 1

        self.keys = []
        self.texts = []
        self.shares = []
        self.beaver_triples = []
        self.multiples = []
        self.squares = []
        self.input_square = []
        self.shares = []
        self.temp = []
        self.input_beaver = []
        self.input_power = []
        self.product = []

    def power_init(self, values, power):
        self.input_power = [[] for _ in range(len(values))]
        for i in range(len(values)):
            self.input_power[i] = [values[i] for _ in range(power)]

    def set_shares(self, shares):
        self.shares.extend(shares)

    def set_keys(self, keys):
        self.keys = keys[:]
        self.keys = reshape_16(self.keys)

    def set_product(self,shares):
        self.product = shares

    def set_texts(self, keys):
        self.texts = keys[:]
        self.texts = reshape_16(self.texts)

    def set_text_row(self, num, text):
        self.texts[num] = text

    def set_beaver_triples(self, beavers):
        self.beaver_triples.extend(beavers)

    def set_squares(self, squares):
        self.squares = squares[:]

    def set_multiples(self, multiples):
        self.multiples = multiples[:]

    def read_file(self, data, file_name=None):
        if not file_name:
            if data == 'key':
                file_name = 'key_P{}.txt'.format(self.compute_no)
            elif data == 'plain':
                file_name = 'plain_P{}.txt'.format(self.compute_no)
            elif data == 'beaver_triple':
                file_name = 'beaver_triple_P{}.txt'.format(self.compute_no)
            elif data == 'multiple':
                file_name = 'multiple_P{}.txt'.format(self.compute_no)
            elif data == 'square':
                file_name = 'sqaure_P{}.txt'.format(self.compute_no)
        with open(file_name,'r') as file:
            length = int(file.readline())
            res = []
            for i in range(length):
                res.append(eval(file.readline()))
            if data == 'key':
                self.set_keys(res)
            elif data == 'plain':
                self.set_texts(res)
            elif data == 'beaver_triple':
                self.set_beaver_triples(res)
            elif data == 'multiple':
                self.set_multiples(res)
            elif data == 'square':
                self.set_squares(res)

    def add_constants(self, constant, value):
        if self.compute_no == 0:
            temp1 = constant + value[0]
        else:
            temp1 = value[0]
        temp2 = constant * self.mac + value[1]
        return (temp1, temp2)

    def add_share_constant(self, s1, s2, constant):
        return (constant*(s1[0]+s2[0]), constant*(s1[1]+s2[1]))

    def generate_mac(self, method='memory'):
        self.mac = gen_rand_gf256()

        if method == 'memory':
            Player.TTP.calculate_mac_sum(self.mac)
        elif method == 'file':
            with open('mac_P{}.txt'.format(self.compute_no), 'w') as file:
                file.write(str(self.mac))

    def beaver_multiply_local(self, multiply_x_mask, multiply_y_mask, triple):
        a, b, c = triple

        if self.compute_no == 0:
            temp1 = a[0] * multiply_y_mask + b[0] * multiply_x_mask + c[0] + multiply_y_mask * multiply_x_mask
        else:
            temp1 = a[0] * multiply_y_mask + b[0] * multiply_x_mask + c[0]
        temp2 = a[1] * multiply_y_mask + b[1] * multiply_x_mask + c[1] + multiply_y_mask * multiply_x_mask*self.mac
        product = (temp1, temp2)
        return product

    # multiply_mask -- (x1-a1, y1-b1,x2-a2,y2-b2,...)
    # triple -- (([a1],[b1],[c1]), ([a2],[b2],[c2]), ...)
    def beaver_multiply_parallel(self, multiply_mask,  triple):
        assert (2 * len(triple) == len(multiply_mask))
        res = []
        for i in range(len(triple)):
            product = self.beaver_multiply_local(multiply_mask[2 * i], multiply_mask[2* i + 1], triple[i])
            res.append(product)
        return res

    def set_multiply_mask(self, value):
        self.multiply_multiply_mask = value[:]

    # suitable to compute one-variable poly
    def poly_multiple_local(self, constants, powers, global_z, multiple, eff_dic=None):
        #t0 = time.time()
        # calculate z^i
        z_powers = [global_z]
        max_power = max(powers)
        for i in range(max_power):
            z_powers.append(z_powers[i] * global_z)
        # calculate coefficient(containing constant and combination)
        if not eff_dic:
            eff_dic = gen_comb_eff(powers)
        rank = [GF256(0)] * max_power
        for i in range(max_power):
            for j in range(len(constants)):
                if powers[j] >= i:
                    if (i, powers[j]) in eff_dic: # characteristic = 2
                        #rank[i] += GF256(constants[j]) * power(global_z, powers[j] - i)
                        rank[i] += GF256(constants[j]) * z_powers[powers[j] - i]
        if self.compute_no == 0:
            res1 = rank[0]
        else:
            res1 = GF256(0)
        res2 = rank[0] * self.mac

        for i in range(1, max_power):
            res1 += multiple[i-1][0] * rank[i]
            res2 += multiple[i-1][1] * rank[i]
        #t1 = time.time()
        #ComputePlayer.multiple_time += t1-t0
        return (res1, res2)

    def poly_multiple_parallel(self, constants, powers, multiples, eff_dic=None):
        t0 = time.time()
        global_value = self.after_broadcast()
        res = []
        if not eff_dic:
            eff_dic = gen_comb_eff(powers)
        for i in range(len(global_value)):
            res.append(self.poly_multiple_local(constants, powers, global_value[i],multiples[i], eff_dic))
        t1 = time.time()
        ComputePlayer.multiple_time += t1 - t0
        return res

    def gen_input_square(self, global_z, square, degree):
        assert (len(square) == degree+1)
        self.input_square.append([])
        self.input_square[-1].append(global_z)
        for i in range(degree):
            self.input_square[-1].append(self.input_square[-1][-1]*self.input_square[-1][-1])
        #print(len(self.input_square[-1]))
        for i in range(degree+1):
            self.input_square[-1][i] = self.add_constants(self.input_square[-1][i], square[i])

    def gen_input_square_parallel(self, square, degree, max_power):
        self.input_square = []
        global_z = self.after_broadcast()
        self.power_init(global_z, max_power)
        assert (len(global_z) == len(square))
        for i in range(len(global_z)):
            self.gen_input_square(global_z[i], square[i], degree)
            for j in range(degree+1):
                self.input_power[i][pow(2, j)-1] = self.input_square[-1][j]



class TrustedThirdPlayer(Player):
    def __init__(self, ip='localhost', rec_port=5000):
        super().__init__(ip, rec_port)
        self.mac_sum = GF256(0)
        self.set_ttp(self)

    def calculate_mac_sum(self, value):
        if type(value) == GF256:
            self.mac_sum+= value
        else:
            with open(value, 'r') as file:
                self.mac_sum += GF256(int(file.readline()))

    def generate_mac_share(self, number, method='memory'):
        all_shares = [[] for _ in range(ComputePlayer.ComputeNum)]
        for i in range(number):
            num = gen_rand_gf256()
            mac_num = num * self.mac_sum
            temp = (num, mac_num)
            shares = self.calculate_share(temp)

            for j in range(ComputePlayer.ComputeNum):
                all_shares[j].append(tuple(shares[j]))
        if method == 'memory':
            for i in range(ComputePlayer.ComputeNum):
                ComputePlayer.ComputeList[i].set_shares(all_shares[i])
        elif method == 'file':
            for i in range(ComputePlayer.ComputeNum):
                with open('mac_share_P{}.txt'.format(i), 'w') as file:
                    print(len(all_shares[i]))
                    file.write(str(len(all_shares[i])))
                    file.write('\n')
                    for j in all_shares[i]:
                        file.write(str(j))
                        file.write('\n')
        else:
            return all_shares

    def generate_squares(self, degree, repeat, storage='memory'):
        all_shares = [[] for _ in range(ComputePlayer.ComputeNum)]
        for i in range(repeat):
            square_loop = []
            temp = gen_rand_gf256()
            mac_num = temp * self.mac_sum
            square_loop.append((temp, mac_num))
            for j in range(1, degree):
                temp = square_loop[j-1][0] * square_loop[j-1][0]
                mac_num = temp * self.mac_sum
                square_loop.append((temp, mac_num))
            res = self.calculate_share_mac(square_loop)
            for j in range(ComputePlayer.ComputeNum):
                all_shares[j].append(res[j])
        if storage == 'memory':
            for i in range(ComputePlayer.ComputeNum):
                ComputePlayer.ComputeList[i].set_squares(all_shares[i])
        elif storage == 'file':
            for i in range(ComputePlayer.ComputeNum):
                with open('square_P{}.txt'.format(i), 'w') as file:
                    print(len(all_shares[i]))
                    file.write(str(len(all_shares[i])))
                    file.write('\n')
                    for j in all_shares[i]:
                        file.write(str(j))
                        file.write('\n')

    def generate_beaver_triples(self, number, method='memory'):
        all_shares = [[] for _ in range(ComputePlayer.ComputeNum)]
        for i in range(number):
            a = gen_rand_gf256()
            b = gen_rand_gf256()
            c = a * b
            temp = (a, b, c, a * self.mac_sum, b*self.mac_sum, c*self.mac_sum)
            shares = self.calculate_share(temp)

            for j in range(ComputePlayer.ComputeNum):
                beaver_mac = tuple((shares[j][k], shares[j][k+3]) for k in range(3))
                all_shares[j].append(beaver_mac)
                #all_shares[j].append(tuple(shares[j][:3]))
        if method == 'memory':
            for i in range(ComputePlayer.ComputeNum):
                ComputePlayer.ComputeList[i].set_beaver_triples(all_shares[i])
        elif method == 'file':
            for i in range(ComputePlayer.ComputeNum):
                with open('beaver_triple_P{}.txt'.format(i), 'w') as file:
                    print(len(all_shares[i]))
                    file.write(str(len(all_shares[i])))
                    file.write('\n')
                    for j in all_shares[i]:
                        file.write(str(j))
                        file.write('\n')

    # method 0 --  repeat eg, (x,y,z) ->(x^2, y^2, ...)
    # method 1 --  no repeat eg,(x,y,z)-> (xy,yz,xz,xyz)
    def generate_multiple(self, number,  degree, repeat, method=0, storage='memory'):
        all_shares = [[] for _ in range(ComputePlayer.ComputeNum)]
        for i in range(repeat):
            secures = []

            for j in range(number):
                secures.append(gen_rand_gf256())
            tmp = secures[:] # record the value of degree i
            res = secures[:]  # record the value of all degrees
            for d in range(1, degree):
                if method == 0:
                    #res = [functools.reduce(lambda x, y: x*y, i) for i in combinations_with_replacement(secures, d)]
                    tmp = [x * y for x in tmp for y in secures]
                    res.extend(tmp)
                else:
                    res = [functools.reduce(lambda x, y: x*y, i) for i in combinations(secures, d)]

            res = [(i, i * self.mac_sum) for i in res]
            res_share = self.calculate_share_mac(res)

            for i in range(ComputePlayer.ComputeNum):
                all_shares[i].append(res_share[i])
        if storage == 'memory':
            for i in range(ComputePlayer.ComputeNum):
                ComputePlayer.ComputeList[i].set_multiples(all_shares[i])
        elif storage == 'file':
            for i in range(ComputePlayer.ComputeNum):
                with open('multiple_P{}.txt'.format(i), 'w') as file:
                    file.write(str(len(all_shares[i])))
                    file.write('\n')
                    for j in all_shares[i]:
                        file.write(str(j))
                        file.write('\n')


class InputTTP(InputPlayer, TrustedThirdPlayer):
    def __init__(self, ip='localhost', rec_port=5000):
        super().__init__(ip, rec_port)


if __name__ == '__main__':
    #print(gen_comb_eff([i for i in range(255)]))
    #print(generate_comb_eff([0, 127, 191, 223, 239, 247, 251, 253, 254]))
    a = TrustedThirdPlayer(rec_port=4000)
    b = InputPlayer(rec_port=10000)
    players = [ComputePlayer(rec_port=5000), ComputePlayer(rec_port=6000)]
    for p in players:
        p.generate_mac()
    b.generate_keys()
    a.generate_squares(8, 100, storage='file')
    a.generate_beaver_triples(10)
    a.generate_multiple(1, 254, 10)
    print(len(players[0].multiples[0][0]))
    multiply_mask = [gen_rand_gf256() for i in range(20)]
    print(players[0].beaver_multiply_parallel(multiply_mask, players[0].beaver_triples))