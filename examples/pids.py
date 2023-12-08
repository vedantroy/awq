from collections import defaultdict

def test(
        M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N,
        GROUP_SIZE_M,
):
    def cdiv(a, b):
        return -(-a // b)

    max_pid = cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)
    gid_to_pid = defaultdict(list)
    pid_to_info = {}
    for pid in range(max_pid):
        # Calculate the various values as per the provided code
        num_pid_m = cdiv(M, BLOCK_SIZE_M)
        num_pid_n = cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        pid_to_info[pid] = dict(
            pid=pid,
            group_id=group_id,
            pid_m=pid_m,
            pid_n=pid_n,
        )
        gid_to_pid[group_id].append(pid)
        # pid_tuples.append((pid, pid_m, pid_n))
    return pid_to_info, gid_to_pid


def test_128M_128N_splitm2_splitn4_mgroup1():
    # Constants
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 128 // 2, 128 // 4
    GROUP_SIZE_M = 1
    prog_infos, gid2pid = test(
        M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N,
        GROUP_SIZE_M,
    )
    assert max(prog_infos['group_id'] for prog_infos in prog_infos.values()) == 1
    assert gid2pid[0] == [0, 1, 2, 3]
    assert gid2pid[1] == [4, 5, 6, 7]

def test_128M_128N_splitm4_splitn2_mgroup2():
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 128 // 4, 128 // 2
    GROUP_SIZE_M = 2
    prog_infos, gid2pid = test(
        M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N,
        GROUP_SIZE_M,
    )
    assert max(prog_infos['group_id'] for prog_infos in prog_infos.values()) == 1
    assert gid2pid[0] == [0, 1, 2, 3]
    assert gid2pid[1] == [4, 5, 6, 7]

    p0, p1, p2, p3 = prog_infos[0], prog_infos[1], prog_infos[2], prog_infos[3]
    assert p0['pid_m']  == 0 and p0['pid_n'] == 0
    assert p1['pid_m']  == 1 and p1['pid_n'] == 0
    assert p2['pid_m']  == 0 and p2['pid_n'] == 1
    assert p3['pid_m']  == 1 and p3['pid_n'] == 1

    p4, p5, p6, p7 = prog_infos[4], prog_infos[5], prog_infos[6], prog_infos[7]
    assert p4['pid_m']  == 2 and p4['pid_n'] == 0
    assert p5['pid_m']  == 3 and p5['pid_n'] == 0
    assert p6['pid_m']  == 2 and p6['pid_n'] == 1
    assert p7['pid_m']  == 3 and p7['pid_n'] == 1

if __name__ == '__main__':
    test_128M_128N_splitm2_splitn4_mgroup1()
    test_128M_128N_splitm4_splitn2_mgroup2()