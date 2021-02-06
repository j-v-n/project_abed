#!/usr/bin/env python3
import os
import random
import argparse
import logging
import numpy as np
import pickle

# from tensorboardX import SummaryWriter

import model

from utils import bleu, dialogues

import torch
import torch.optim as optim
import torch.nn.functional as F

import mlflow


SAVES_DIR = "saves"

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHES = 100

log = logging.getLogger("train")

TEACHER_PROB = 0.5
saves_path = "./"


def run_test(test_data, net, end_token, device="cpu"):
    bleu_sum = 0.0
    bleu_count = 0
    for p1, p2 in test_data:
        input_seq = model.pack_input(p1, net.emb, device)
        enc = net.encode(input_seq)
        _, tokens = net.decode_chain_argmax(
            enc,
            input_seq.data[0:1],
            seq_len=dialogues.MAX_TOKENS,
            stop_at_token=end_token,
        )
        bleu_sum += bleu.calc_bleu(tokens, p2[1:])
        bleu_count += 1
    return bleu_sum / bleu_count


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO
    )

    device = torch.device("cuda")
    print("checking if phrase pairs exist")

    missing = 0
    if os.path.isfile("phrase_pairs.dat"):
        print("phrase_pairs and emb_dict exist, therefore loading them ...")

        phrase_pairs = dialogues.load_phrase_pairs("./")
        emb_dict = dialogues.load_emb_dict("./")

    else:

        print("phrase_pairs and emb_dict do not exist, therefore creating them ...")
        phrase_pairs, emb_dict = dialogues.load_data(
            name="train",
            max_tokens=dialogues.MAX_TOKENS,
            min_token_freq=dialogues.MIN_TOKEN_FREQ,
        )

        print("saving phrase pairs and emb_dict for future use")
        dialogues.save_emb_dict("./", emb_dict)
        dialogues.save_phrase_pairs("./", phrase_pairs)

    log.info(
        "Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict)
    )
    # data.save_emb_dict(saves_path, emb_dict)

    end_token = emb_dict[dialogues.END_TOKEN]

    train_data = dialogues.encode_phrase_pairs(phrase_pairs, emb_dict)

    rand = np.random.RandomState(dialogues.SHUFFLE_SEED)

    rand.shuffle(train_data)

    log.info("Training data converted, got %d samples", len(train_data))

    test_phrase_pairs, _ = dialogues.load_data(
        name="test",
        max_tokens=dialogues.MAX_TOKENS,
        min_token_freq=dialogues.MIN_TOKEN_FREQ,
    )

    log.info("Obtained %d test phrase pairs", len(test_phrase_pairs), len(emb_dict))

    test_data = dialogues.encode_phrase_pairs(test_phrase_pairs, emb_dict)

    # train_data, test_data = data.split_train_test(train_data)
    # log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))

    net = model.PhraseModel(
        emb_size=model.EMBEDDING_DIM,
        dict_size=len(emb_dict),
        hid_size=model.HIDDEN_STATE_SIZE,
    ).to(device)

    log.info("Model: %s", net)

    # writer = SummaryWriter(comment="-" + args.name)

    optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    best_bleu = None
    with mlflow.start_run():
        for epoch in range(MAX_EPOCHES):
            losses = []
            bleu_sum = 0.0
            bleu_count = 0
            for batch in dialogues.iterate_batches(train_data, BATCH_SIZE):
                optimiser.zero_grad()
                input_seq, out_seq_list, _, out_idx = model.pack_batch(
                    batch, net.emb, device
                )
                enc = net.encode(input_seq)

                net_results = []
                net_targets = []
                for idx, out_seq in enumerate(out_seq_list):
                    ref_indices = out_idx[idx][1:]
                    enc_item = net.get_encoded_item(enc, idx)
                    if random.random() < TEACHER_PROB:
                        r = net.decode_teacher(enc_item, out_seq)
                        bleu_sum += model.seq_bleu(r, ref_indices)
                    else:
                        r, seq = net.decode_chain_argmax(
                            enc_item, out_seq.data[0:1], len(ref_indices)
                        )
                        bleu_sum += bleu.calc_bleu(seq, ref_indices)
                    net_results.append(r)
                    net_targets.extend(ref_indices)
                    bleu_count += 1
                results_v = torch.cat(net_results)
                targets_v = torch.LongTensor(net_targets).to(device)
                loss_v = F.cross_entropy(results_v, targets_v)
                loss_v.backward()
                optimiser.step()

                losses.append(loss_v.item())
            bleu_average = bleu_sum / bleu_count
            bleu_test = run_test(test_data, net, end_token, device)
            log.info(
                "Epoch %d: mean loss %.3f, mean BLEU %.3f, test BLEU %.3f",
                epoch,
                np.mean(losses),
                bleu_average,
                bleu_test,
            )
            mlflow.log_metric("loss", np.mean(losses))
            mlflow.log_metric("bleu_score", bleu_average)
            mlflow.log_metric("bleu_test", bleu_test)

            # writer.add_scalar("loss", np.mean(losses), epoch)
            # writer.add_scalar("bleu", bleu, epoch)
            # writer.add_scalar("bleu_test", bleu_test, epoch)
            if best_bleu is None or best_bleu < bleu_test:
                if best_bleu is not None:
                    out_name = os.path.join(
                        saves_path, "pre_bleu_%.3f_%02d.dat" % (bleu_test, epoch)
                    )
                    torch.save(net.state_dict(), out_name)
                    log.info("Best BLEU updated %.3f", bleu_test)
                best_bleu = bleu_test

            if epoch % 10 == 0:
                out_name = os.path.join(
                    saves_path,
                    "epoch_%03d_%.3f_%.3f.dat" % (epoch, bleu_average, bleu_test),
                )
                torch.save(net.state_dict(), out_name)

    # writer.close()
