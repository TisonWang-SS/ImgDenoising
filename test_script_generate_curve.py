import csv

# test script to generate metric curves during training

log_path = './train_log.log'
out_file = './train_loss.csv'

def main():
    train_loss = []
    valid_loss = []
    with open(log_path, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith('#####Train#####'):
                epoch = line.split('Epoch: ')[1].split(',')[0]
                UNetR_Loss = line.split('UNetR_Loss: ')[1].split(',')[0]
                UNetG_Loss = line.split('UNetG_Loss: ')[1].split(',')[0]
                Discriminator_Loss = line.split('Discriminator_Loss: ')[1].split(',')[0]
                train_loss.append((epoch, UNetR_Loss, UNetG_Loss, Discriminator_Loss))
            elif line.startswith('#####Valid#####'):
                epoch = line.split('Epoch: ')[1].split(',')[0]
                mae_loss = line.split('mae_loss: ')[1].split(',')[0]
                psnr_loss = line.split('psnr_loss: ')[1].split(',')[0]
                ssim_loss = line.split('ssim_loss: ')[1].split(',')[0]
                valid_loss.append((epoch, mae_loss, psnr_loss, ssim_loss))
            line = f.readline()
    
    with open(out_file, 'w', newline='') as output:
        csv_out = csv.writer(output)
        csv_out.writerow(['Epoch', 'train_UNetR_Loss', 'train_UNetG_Loss', 'train_Discriminator_Loss', 'valid_mae_loss', 'valid_psnr_loss', 'valid_ssim_loss'])
        for i in range(len(train_loss)):
            csv_out.writerow((train_loss[i][0], train_loss[i][1], train_loss[i][2], train_loss[i][3], valid_loss[i][1], valid_loss[i][2], valid_loss[i][3]))


if __name__ == '__main__':
    main()