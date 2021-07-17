from player import *
import datetime
from threading import Thread


def process_number(process):
    if process == 0:
        return 16
    else:
        return 4


def broadcast():
    players = ComputePlayer.ComputeList
    #print(players[0].rec_port, players[1].rec_port)
    t1 = Thread(target=players[0].prep_rec, args=())
    t2 = Thread(target=players[1].send_num, args=(players[1].broadcast, players[0].ip, players[0].rec_port))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    #print(players[0].rec_port, players[1].rec_port)
    t1 = Thread(target=players[1].prep_rec, args=())
    t2 = Thread(target=players[0].send_num, args=(players[0].broadcast, players[1].ip, players[1].rec_port))
    t1.start()
    t2.start()
    t1.join()
    t2.join()


# P0 send P0.broadcast to P1
def send(P0, P1):
    t1 = Thread(target=P1.prep_rec, args=())
    t2 = Thread(target=P0.send_num, args=(P0.broadcast, P1.ip, P1.rec_port))
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def pop(lis, num):
    tmp = lis[: num]
    del lis[: num]
    return tmp


# process: 0-- sub_byte, 1--key_expansion
def poly_multiple(players, constants, powers, process):
    if len(constants) != len(powers):
        raise RuntimeError("No equal size")
    for p in players:
        if process == 0:
            p.current_multiple = pop(p.multiples, 16)
            p.temp = []

            for i in range(4):
                for j in range(4):
                    p.temp.append(p.texts[i][j][0]-p.current_multiple[4*i+j][0][0])
            p.set_broadcast(p.temp)
        else:
            p.current_multiple = pop(p.multiples, 4)
            p.set_broadcast(p.keys[i][3][0]-p.current_multiple[i][0][0] for i in range(4))
    broadcast()
    for p in players:
        res = p.poly_multiple_parallel(constants, powers, p.current_multiple)
        if process == 0:
            p.set_texts(res)
        else:
            p.set_keys_column(3, res)


def sbox(players, process):
    constants = [
        0x63, 0x8F, 0xB5, 0x01, 0xF4, 0x25, 0xF9, 0x09, 0x05
    ]
    powers = [
        0, 127, 191, 223, 239, 247, 251, 253, 254
    ]
    poly_multiple(players, constants, powers, process)


def generate_sbox_input_square(players, process, degree=7, max_power=254):
    for p in players:
        if process == 0:
            p.current_square = pop(p.squares, 16)
            p.temp = []
            for i in range(4):
                for j in range(4):
                    p.temp.append(p.texts[i][j][0] - p.current_square[4*i+j][0][0])
            p.set_broadcast(p.temp)
        else:
            p.current_square = pop(p.squares, 4)
            p.set_broadcast([p.keys[i][3][0] - p.current_square[i][0][0] for i in range(4)])
    broadcast()
    for p in players:
        p.gen_input_square_parallel(p.current_square, degree, max_power)


def beaver_multiplication(players, save_pos, process):
    for p in players:
        num = len(p.input_beaver)//2
        assert num == len(save_pos) * process_number(process)
        p.current_beaver_triple = pop(p.beaver_triples, num)
        p.temp = []
        for i in range(num):
            p.temp.append(p.input_beaver[2*i]-p.current_beaver_triple[i][0][0])
            p.temp.append(p.input_beaver[2*i+1]-p.current_beaver_triple[i][1][0])
        p.set_broadcast(p.temp)
    broadcast()
    for p in players:
        p.product = p.beaver_multiply_parallel(p.after_broadcast(), p.current_beaver_triple)
        for i in range(process_number(process)):
            for j in range(len(save_pos)):
                p.input_power[i][save_pos[j]] = p.product[len(save_pos)*i+j]


def sbox_sqaure_optimized(players, process):
    constants = [
        0x63, 0x8F, 0xB5, 0x01, 0xF4, 0x25, 0xF9, 0x09, 0x05
    ]
    powers = [0, 127, 191, 223, 239, 247, 251, 253, 254]
    generate_sbox_input_square(players, process)
    for p in players:
        p.length = len(p.input_square)
        p.input_beaver = []
        for i in range(p.length):
            p.input_beaver.extend([p.input_power[i][0][0], p.input_power[i][1][0],
                                   p.input_power[i][63][0], p.input_power[i][127][0]])
    beaver_multiplication(players, [2, 191], process)
    for p in players:
        p.input_beaver = []
        for i in range(p.length):
            p.input_beaver.extend([p.input_power[i][2][0], p.input_power[i][3][0],
                                   p.input_power[i][31][0], p.input_power[i][191][0]])
    beaver_multiplication(players, [6, 223], process)
    for p in players:
        p.input_beaver = []
        for i in range(p.length):
            p.input_beaver.extend([p.input_power[i][6][0], p.input_power[i][7][0],
                                   p.input_power[i][15][0], p.input_power[i][223][0]])
    beaver_multiplication(players, [14, 239], process)
    for p in players:
        p.input_beaver = []
        for i in range(p.length):
            p.input_beaver.extend([p.input_power[i][14][0], p.input_power[i][15][0],
                                   p.input_power[i][7][0], p.input_power[i][239][0],
                                   p.input_power[i][14][0], p.input_power[i][223][0],
                                   p.input_power[i][6][0], p.input_power[i][239][0]])
    beaver_multiplication(players, [30, 247, 238, 246], process)
    for p in players:
        p.input_beaver = []
        for i in range(p.length):
            p.input_beaver.extend([p.input_power[i][30][0], p.input_power[i][31][0],
                                   p.input_power[i][3][0], p.input_power[i][247][0],
                                   p.input_power[i][30][0], p.input_power[i][191][0],
                                   p.input_power[i][2][0], p.input_power[i][247][0]])
    beaver_multiplication(players, [62, 251, 222, 250], process)
    for p in players:
        p.input_beaver = []
        for i in range(p.length):
            p.input_beaver.extend([p.input_power[i][62][0], p.input_power[i][63][0],
                                   p.input_power[i][1][0], p.input_power[i][251][0],
                                   p.input_power[i][62][0], p.input_power[i][127][0],
                                   p.input_power[i][0][0], p.input_power[i][251][0]])
    beaver_multiplication(players, [126, 253, 190, 252], process)
    for p in players:
        if p.compute_no == 0:
            p.temp = [(GF256(constants[0]), GF256(constants[0])*p.mac) for _ in range(process_number(process))]
        else:
            p.temp = [(GF256(0), GF256(constants[0])*p.mac) for _ in range(process_number(process))]
        for i in range(process_number(process)):
            for j in range(1, len(constants)):
                p.temp[i] = p.add_share_constant(p.temp[i], p.input_power[i][powers[j]-1], GF256(constants[j]))
        if process == 0:
            p.set_texts(p.temp)
        else:
            p.set_key_column(3, p.temp)


def sbox_inv_multiple(players, process):
    constants = [
        0x52, 0xF3, 0x7E, 0x1E, 0x90, 0xBB, 0x2C, 0x8A, 0x1C, 0x85, 0x6D, 0xC0, 0xB2, 0x1B, 0x40, 0x23,
        0xF6, 0x73, 0x29, 0xD9, 0x39, 0x21, 0xCF, 0x3D, 0x9A, 0x8A, 0x2F, 0xCF, 0x7B, 0x04, 0xE8, 0xC8,
        0x85, 0x7B, 0x7C, 0xAF, 0x86, 0x2F, 0x13, 0x65, 0x75, 0xD3, 0x6D, 0xD4, 0x89, 0x8E, 0x65, 0x05,
        0xEA, 0x77, 0x50, 0xA3, 0xC5, 0x01, 0x0B, 0x46, 0xBF, 0xA7, 0x0C, 0xC7, 0x8E, 0xF2, 0xB1, 0xCB,
        0xE5, 0xE2, 0x10, 0xD1, 0x05, 0xB0, 0xF5, 0x86, 0xE4, 0x03, 0x71, 0xA6, 0x56, 0x03, 0x9E, 0x3E,
        0x19, 0x18, 0x52, 0x16, 0xB9, 0xD3, 0x38, 0xD9, 0x04, 0xE3, 0x72, 0x6B, 0xBA, 0xE8, 0xBF, 0x9D,
        0x1D, 0x5A, 0x55, 0xFF, 0x71, 0xE1, 0xA8, 0x8E, 0xFE, 0xA2, 0xA7, 0x1F, 0xDF, 0xB0, 0x03, 0xCB,
        0x08, 0x53, 0x6F, 0xB0, 0x7F, 0x87, 0x8B, 0x02, 0xB1, 0x92, 0x81, 0x27, 0x40, 0x2E, 0x1A, 0xEE,
        0x10, 0xCA, 0x82, 0x4F, 0x09, 0xAA, 0xC7, 0x55, 0x24, 0x6C, 0xE2, 0x58, 0xBC, 0xE0, 0x26, 0x37,
        0xED, 0x8D, 0x2A, 0xD5, 0xED, 0x45, 0xC3, 0xEC, 0x1C, 0x3E, 0x2A, 0xB3, 0x9E, 0xB7, 0x38, 0x82,
        0x23, 0x2D, 0x87, 0xEA, 0xDA, 0x45, 0x24, 0x03, 0xE7, 0xC9, 0xE3, 0xD3, 0x4E, 0xDD, 0x11, 0x4E,
        0x81, 0x91, 0x91, 0x59, 0xA3, 0x80, 0x92, 0x7E, 0xDB, 0xC4, 0x20, 0xEC, 0xDB, 0x55, 0x7F, 0xA8,
        0xC1, 0x64, 0xAB, 0x1B, 0xFD, 0x60, 0x05, 0x13, 0x2C, 0xA9, 0x76, 0xA5, 0x1D, 0x32, 0x8E, 0x1E,
        0xC0, 0x65, 0xCB, 0x8B, 0x93, 0xE4, 0xAE, 0xBE, 0x5F, 0x2C, 0x3B, 0xD2, 0x0F, 0x9F, 0x42, 0xCC,
        0x6C, 0x80, 0x68, 0x43, 0x09, 0x23, 0xC5, 0x6D, 0x1D, 0x18, 0xBD, 0x5E, 0x1B, 0xB4, 0x85, 0x49,
        0xBC, 0x0D, 0x1F, 0xA6, 0x6B, 0xD8, 0x22, 0x01, 0x7A, 0xC0, 0x55, 0x16, 0xB3, 0xCF, 0x05
    ]
    powers = [i for i in range(255)]
    poly_multiple(players, constants, powers, process)


if __name__ == '__main__':
    #print(gen_comb_eff([i for i in range(255)]))
    #print(generate_comb_eff([0, 127, 191, 223, 239, 247, 251, 253, 254]))

    a = TrustedThirdPlayer(rec_port=4000)
    b = InputPlayer(rec_port=10000)
    players = [ComputePlayer(rec_port=23000), ComputePlayer(rec_port=24000)]
    for p in players:
        p.generate_mac()
    b.generate_keys()
    a.generate_squares(8, 100)
    a.generate_beaver_triples(1000)
    a.generate_multiple(1, 254, 100)
    b.generate_keys()
    b.generate_texts()
    constants = [
        0x63, 0x8F, 0xB5, 0x01, 0xF4, 0x25, 0xF9, 0x09, 0x05
    ]
    powers = [
        0, 127, 191, 223, 239, 247, 251, 253, 254
    ]
    poly_multiple(players, constants, powers, 0)
    generate_sbox_input_square(players, 0)
    sbox_sqaure_optimized(players, 0)

