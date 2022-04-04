class LabelMapper():
    @staticmethod
    def get_cifar10_mapper():
        """
        returns mapper for cifar10 dataset
        """
        result_mapper = dict(zip(list(range(10)),
        ['airplane','automobile','bird','cat','deer',
        'dog','frog','horse','ship','truck']))
        return result_mapper