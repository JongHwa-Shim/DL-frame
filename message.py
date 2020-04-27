def leave_log (losses, accuracy_list, epoch, mode):
    loss = sum(losses)/len(losses)
    accuracy = accuracy_list.count(1)/len(accuracy_list)

    if mode=='train':
        base = ("Epoch: {epoch:d}  Train_Loss: {loss:.8}  Train_Accuracy: {accuracy:.8}")
    else:
        base = ("Epoch: {epoch:d}  Valid_Loss: {loss:.8}  Valid_Accuracy: {accuracy:.8}")

    message = base.format(epoch=epoch, loss=loss, accuracy=accuracy)
    return message, loss, accuracy