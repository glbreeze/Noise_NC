from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from numpy.testing import assert_array_almost_equal
from typing import Any, Callable, Optional, Tuple

np.random.seed(15)


def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.), "Noise rate must be between 0 and 1"

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class


class cifar10Noisy(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noisy_rate=0.0, asym=False):
        super(cifar10Noisy, self).__init__(
            root, download=download, transform=transform, 
            target_transform=target_transform, train=train
            )
        self.clean_targets = self.targets.copy()
        self.train = train
        
        if not self.train:
            print("Test mode: keep clean labels")
            return
        
        print(f"Training mode: applying noise (rate={noisy_rate}, asym={asym})")
        
        if asym: # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(noisy_rate * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
            return
        elif noisy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(noisy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy %d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                print(f"Number of (noisy) samples in class {i}: {n_noisy}")
            return

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target, target_clean = self.data[index], self.targets[index], self.clean_targets[index]
        
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target, target_clean


class cifar100Noisy(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noisy_rate=0.0, asym=False, seed=0):
        super(cifar100Noisy, self).__init__(root, download=download, transform=transform, target_transform=target_transform, train=train)
        self.clean_targets = self.targets.copy()
        self.train = train
        
        if not self.train:
            print("Test mode: keep clean labels")
            return
        
        print(f"Training mode: applying noise (rate={noisy_rate}, asym={asym})")
        
        if asym: # mistakes are inside the same superclass of 10 classes, e.g. 'fish'
            nb_classes = 100
            P = np.eye(nb_classes)
            n = noisy_rate
            nb_superclasses = 20
            nb_subclasses = 5

            if n > 0.0:
                for i in np.arange(nb_superclasses):
                    init, end = i * nb_subclasses, (i+1) * nb_subclasses
                    P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

                    y_train_noisy = multiclass_noisify(np.array(self.targets), P=P, random_state=seed)
                    actual_noise = (y_train_noisy != np.array(self.targets)).mean()
                assert actual_noise > 0.0
                print('Actual noise %.2f' % actual_noise)
                self.targets = y_train_noisy.tolist()
            return
        
        elif noisy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(noisy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(100)]
            class_noisy = int(n_noisy / 100)
            noisy_idx = []
            for d in range(100):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy %d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=100, current_class=self.targets[i])
                
            print(f"Total noisy indices selected: {len(noisy_idx)}")
            print("Print noisy label generation statistics:")
            for i in range(100):
                n_noisy = np.sum(np.array(self.targets) == i)
                print(f"Class {i}, number of noisy {n_noisy}")
            return
        
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        
        img, target, target_clean = self.data[index], self.targets[index], self.clean_targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, target_clean


class DatasetGenerator():
    def __init__(self, batchSize=128, eval_batch_size=256, dataPath='../../datasets',
                 seed=123, numOfWorkers=4, asym=False, dataset_type='cifar10',
                 cutout_length=16, noise_rate=0.4):
        self.seed = seed
        np.random.seed(seed)
        self.batchSize = batchSize
        self.eval_batch_size = eval_batch_size
        self.dataPath = dataPath
        self.numOfWorkers = numOfWorkers
        self.cutout_length = cutout_length
        self.noise_rate = noise_rate
        self.dataset_type = dataset_type
        self.asym = asym
        self.data_loaders = self.loadData()
        return

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        if self.dataset_type == 'cifar100':
            CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
            CIFAR_STD = [0.2673, 0.2564, 0.2762]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            train_dataset = cifar100Noisy(root=self.dataPath,
                                          train=True,
                                          transform=train_transform,
                                          download=True,
                                          asym=self.asym,
                                          seed=self.seed,
                                          noisy_rate=self.noise_rate)

            test_dataset = cifar100Noisy(root=self.dataPath,
                                          train=False,
                                          transform=test_transform,
                                          download=True,
                                          asym=self.asym,
                                          seed=self.seed,
                                          noisy_rate=self.noise_rate)

        elif self.dataset_type == 'cifar10':
            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            train_dataset = cifar10Noisy(root=self.dataPath, train=True,
                                         transform=train_transform, download=True,
                                         asym=self.asym, noisy_rate=self.noise_rate)

            test_dataset = cifar10Noisy(root=self.dataPath, train=False,
                                         transform=test_transform, download=True,
                                         asym=self.asym, noisy_rate=self.noise_rate)
        else:
            raise("Unknown Dataset")

        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.batchSize,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.numOfWorkers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.numOfWorkers)

        print("Num of train %d" % (len(train_dataset)))
        print("Num of test %d" % (len(test_dataset)))

        return data_loaders
