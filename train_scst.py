#!/usr/bin/env python3
import os
import random
import argparse
import logging
import numpy as np

import model


from utils import bleu, dialogues

import torch
import torch.optim as optim
import torch.nn.functional as F

import mlflow
from tqdm import tqdm

SAVES_DIR = "saves"
saves_path = "./"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHES = 10000

log = logging.getLogger("train")


def run_test(test_data, net, end_token, device="cpu"):
    bleu_sum = 0.0
    bleu_count = 0
    for p1, p2 in test_data:
        input_seq = model.pack_input(p1, net.emb, device)
        enc = net.encode(input_seq)
        _, tokens = net.decode_chain_argmax(
            enc, input_seq.data[0:1], seq_len=data.MAX_TOKENS, stop_at_token=end_token
        )
        ref_indices = [indices[1:] for indices in p2]
        bleu_sum += bleu.calc_bleu_many(tokens, ref_indices)
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

    end_token = emb_dict[dialogues.END_TOKEN]

    train_data = dialogues.encode_phrase_pairs(phrase_pairs, emb_dict)

    rand = np.random.RandomState(dialogues.SHUFFLE_SEED)

    rand.shuffle(train_data)

    test_phrase_pairs, _ = dialogues.load_data(
        name="test",
        max_tokens=dialogues.MAX_TOKENS,
        min_token_freq=dialogues.MIN_TOKEN_FREQ,
    )

    log.info("Obtained %d test phrase pairs", len(test_phrase_pairs), len(emb_dict))

    test_data = dialogues.encode_phrase_pairs(test_phrase_pairs, emb_dict)

    train_data = dialogues.group_train_data(train_data)
    test_data = dialogues.group_train_data(test_data)
    log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))

    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

    net = model.PhraseModel(
        emb_size=model.EMBEDDING_DIM,
        dict_size=len(emb_dict),
        hid_size=model.HIDDEN_STATE_SIZE,
    ).to(device)
    log.info("Model: %s", net)

    net.load_state_dict(torch.load("epoch_020_0.174_0.097.dat"))
    log.info("Model loaded, continue training in RL mode...")

    # BEGIN token
    beg_token = torch.LongTensor([emb_dict[dialogues.BEGIN_TOKEN]]).to(device)

    optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    batch_idx = 0
    best_bleu = None

    with mlflow.start_run():
        for epoch in tqdm(range(MAX_EPOCHES)):
            random.shuffle(train_data)
            dial_shown = False

            total_samples = 0
            skipped_samples = 0
            bleus_argmax = []
            bleus_sample = []

            for batch in tqdm(dialogues.iterate_batches(train_data, BATCH_SIZE)):
                batch_idx += 1
                optimiser.zero_grad()
                input_seq, input_batch, output_batch = model.pack_batch_no_out(
                    batch, net.emb, device
                )
                enc = net.encode(input_seq)

                net_policies = []
                net_actions = []
                net_advantages = []
                beg_embedding = net.emb(beg_token)

                for idx, inp_idx in enumerate(input_batch):
                    total_samples += 1
                    ref_indices = [indices[1:] for indices in output_batch[idx]]
                    item_enc = net.get_encoded_item(enc, idx)
                    r_argmax, actions = net.decode_chain_argmax(
                        item_enc,
                        beg_embedding,
                        dialogues.MAX_TOKENS,
                        stop_at_token=end_token,
                    )
                    argmax_bleu = bleu.calc_bleu_many(actions, ref_indices)
                    bleus_argmax.append(argmax_bleu)

                    if argmax_bleu > 0.99:
                        skipped_samples += 1
                        continue

                    if not dial_shown:
                        log.info(
                            "Input: %s",
                            dialogues.untokenize(
                                dialogues.decode_words(inp_idx, rev_emb_dict)
                            ),
                        )
                        ref_words = [
                            dialogues.untokenize(
                                dialogues.decode_words(ref, rev_emb_dict)
                            )
                            for ref in ref_indices
                        ]
                        log.info("Refer: %s", " ~~|~~ ".join(ref_words))
                        log.info(
                            "Argmax: %s, bleu=%.4f",
                            dialogues.untokenize(
                                dialogues.decode_words(actions, rev_emb_dict)
                            ),
                            argmax_bleu,
                        )

                    for _ in range(4):
                        r_sample, actions = net.decode_chain_sampling(
                            item_enc,
                            beg_embedding,
                            dialogues.MAX_TOKENS,
                            stop_at_token=end_token,
                        )
                        sample_bleu = bleu.calc_bleu_many(actions, ref_indices)

                        if not dial_shown:
                            log.info(
                                "Sample: %s, bleu=%.4f",
                                dialogues.untokenize(
                                    dialogues.decode_words(actions, rev_emb_dict)
                                ),
                                sample_bleu,
                            )

                        net_policies.append(r_sample)
                        net_actions.extend(actions)
                        net_advantages.extend(
                            [sample_bleu - argmax_bleu] * len(actions)
                        )
                        bleus_sample.append(sample_bleu)
                    dial_shown = True

                if not net_policies:
                    continue

                policies_v = torch.cat(net_policies)
                actions_t = torch.LongTensor(net_actions).to(device)
                adv_v = torch.FloatTensor(net_advantages).to(device)
                log_prob_v = F.log_softmax(policies_v, dim=1)
                log_prob_actions_v = (
                    adv_v * log_prob_v[range(len(net_actions)), actions_t]
                )
                loss_policy_v = -log_prob_actions_v.mean()

                loss_v = loss_policy_v
                loss_v.backward()
                optimiser.step()

            bleu_test = run_test(test_data, net, end_token, device)
            bleu_average = np.mean(bleus_argmax)

            mlflow.log_metric("bleu_test", bleu_test)
            mlflow.log_metric("bleu_argmax", bleu_average)
            mlflow.log_metric("bleu_sample", np.mean(bleus_sample))
            mlflow.log_metric("skipped_samples", skipped_samples / total_samples)
            mlflow.log_metric("epoch", epoch)

            log.info("Epoch %d, test BLEU: %.3f", epoch, bleu_test)
            if best_bleu is None or best_bleu < bleu_test:
                best_bleu = bleu_test
                log.info("Best bleu updated: %.4f", bleu_test)
                torch.save(
                    net.state_dict(),
                    os.path.join(saves_path, "bleu_%.3f_%02d.dat" % (bleu_test, epoch)),
                )
            if epoch % 5 == 0:
                torch.save(
                    net.state_dict(),
                    os.path.join(
                        saves_path,
                        "epoch_%03d_%.3f_%.3f.dat" % (epoch, bleu, bleu_test),
                    ),
                )

