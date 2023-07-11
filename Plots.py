from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'Times New Roman'})

def plotSEIRS(ss, es, xs, rs, figname=None):
    s_arr, e_arr, x_arr, r_arr = (np.array(ss).squeeze(2), 
                                  np.array(es).squeeze(2), 
                                  np.array(xs).squeeze(2),
                                  np.array(rs).squeeze(2))

    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)
    plt.plot(s_arr)
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$s_i$")

    plt.subplot(2,2,2)
    plt.plot(e_arr)
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$e_i$")

    plt.subplot(2,2,3)
    plt.plot(x_arr)
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$x_i$")

    plt.subplot(2,2,4)
    plt.plot(r_arr)
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$r_i$")

    plt.tight_layout()

    if figname:
        plt.savefig("figures/" + figname + ".png", dpi=300)

def compareSEIRS(ss_1,es_1,xs_1,rs_1,ss_2,es_2,xs_2,rs_2,figname=None,sub_section=None):
    s_arr, e_arr, x_arr, r_arr = (np.array(ss_1).squeeze(2), 
                                  np.array(es_1).squeeze(2), 
                                  np.array(xs_1).squeeze(2),
                                  np.array(rs_1).squeeze(2))
    
    s_arr_p, e_arr_p, x_arr_p, r_arr_p = (np.array(ss_2).squeeze(2), 
                                  np.array(es_2).squeeze(2), 
                                  np.array(xs_2).squeeze(2),
                                  np.array(rs_2).squeeze(2))

    if not sub_section:
        sub_section = [i for i in range(s_arr.shape[1])]

    plt.figure(figsize=(6,3))
    # plt.figure(figsize=(24,12))
    plt.subplot(2,2,1)
    plt.plot(s_arr[:,sub_section])
    plt.gca().set_prop_cycle(None)
    plt.plot(s_arr_p[:,sub_section],"--")
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$s_i$")
    # plt.ylim(-0.1,1.1)

    plt.subplot(2,2,2)
    plt.plot(e_arr[:,sub_section])
    plt.gca().set_prop_cycle(None) 
    plt.plot(e_arr_p[:,sub_section],"--")
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$e_i$")
    # plt.ylim(-0.1,1.1)

    plt.subplot(2,2,3)
    plt.plot(x_arr[:,sub_section])
    plt.gca().set_prop_cycle(None)
    plt.plot(x_arr_p[:,sub_section],"--")
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$x_i$")
    # # plt.ylim(-0.1,1.1)

    plt.subplot(2,2,4)
    plt.plot(r_arr[:,sub_section])
    plt.gca().set_prop_cycle(None)
    plt.plot(r_arr_p[:,sub_section],"--")
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$r_i$")
    # plt.ylim(-0.1,1.1)

    plt.tight_layout()

    if figname:
        plt.savefig("figures/" + figname + ".png", dpi=300)

def compareSEIRS_vert(ss_1,es_1,xs_1,rs_1,ss_2,es_2,xs_2,rs_2,figname=None,sub_section=None):
    s_arr, e_arr, x_arr, r_arr = (np.array(ss_1).squeeze(2), 
                                  np.array(es_1).squeeze(2), 
                                  np.array(xs_1).squeeze(2),
                                  np.array(rs_1).squeeze(2))
    
    s_arr_p, e_arr_p, x_arr_p, r_arr_p = (np.array(ss_2).squeeze(2), 
                                  np.array(es_2).squeeze(2), 
                                  np.array(xs_2).squeeze(2),
                                  np.array(rs_2).squeeze(2))

    if not sub_section:
        sub_section = [i for i in range(s_arr.shape[1])]

    plt.figure(figsize=(4,6))
    # plt.figure(figsize=(24,12))
    plt.subplot(4,1,1)
    plt.plot(s_arr[:,sub_section])
    plt.gca().set_prop_cycle(None)
    plt.plot(s_arr_p[:,sub_section],"--")
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$s_i$")
    # plt.ylim(-0.1,1.1)

    plt.subplot(4,1,2)
    plt.plot(e_arr[:,sub_section])
    plt.gca().set_prop_cycle(None) 
    plt.plot(e_arr_p[:,sub_section],"--")
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$e_i$")
    # plt.ylim(-0.1,1.1)

    plt.subplot(4,1,3)
    plt.plot(x_arr[:,sub_section])
    plt.gca().set_prop_cycle(None)
    plt.plot(x_arr_p[:,sub_section],"--")
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$x_i$")
    # # plt.ylim(-0.1,1.1)

    plt.subplot(4,1,4)
    plt.plot(r_arr[:,sub_section])
    plt.gca().set_prop_cycle(None)
    plt.plot(r_arr_p[:,sub_section],"--")
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$r_i$")
    # plt.ylim(-0.1,1.1)

    plt.tight_layout()

    if figname:
        plt.savefig("figures/" + figname + ".png", dpi=300)