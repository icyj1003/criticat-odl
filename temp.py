 # training
                train_labels = None
                train_outputs = None
                train_loss = 0
                for idx, batch in enumerate(train_loader):
                    # repare inputs and labels for training
                    inputs, labels = repare_input(
                        configs, batch, device
                    )

                    # forward pass
                    loss, outputs = train_one(
                        model,
                        optimizer,
                        criterion,
                        inputs=inputs,
                        targets=labels,
                    )

                    # merging labels and outputs
                    train_labels = (
                        torch.cat([train_labels, labels])
                        if train_labels != None
                        else labels
                    )
                    train_outputs = (
                        torch.cat([train_outputs, torch.argmax(outputs, dim=1)])
                        if train_outputs != None
                        else torch.argmax(outputs, dim=1)
                    )

                    # append loss
                    train_loss += loss.item() / len(train_loader)

                train_acc = accuracy_score(train_labels.cpu(), train_outputs.cpu())
                train_f1 = f1_score(
                    train_labels.cpu(), train_outputs.cpu(), average="macro"
                )