def main():
    with open('eval.csv', 'w') as eval_output_file: 
        for image_id in range(10000):
            eval_output_file.write('{},{}{}{},{},{},{}\n'.format(image_id, "data/tiny-imagenet-200/test/images/test_", image_id, ".JPEG", 64, 64, 3))


if __name__ == '__main__':
    main()
