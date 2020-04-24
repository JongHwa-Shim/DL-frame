def leave_log (losses, accuracy_list, epoch):
    loss = sum(losses)/len(losses)
    accuracy = accuracy_list.count(1)/len(accuracy_list)

    base = ("Epoch: {epoch:d}  Train_Loss: {loss:.8}  Train_Accuracy: {accuracy:.8}\n")
    message = base.format(epoch=epoch, loss=loss, accuracy=accuracy)
    return message, loss, accuracy