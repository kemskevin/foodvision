terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Identifiants lus dans les variables d'environnement :
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (et optionnellement AWS_DEFAULT_REGION)
provider "aws" {
  region = var.region
}

# Dernière image Ubuntu 24.04 officielle (Canonical)
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/*/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_key_pair" "deployer" {
  key_name   = "${var.name}-deployer"
  public_key = file(pathexpand(var.ssh_public_key_path))
}

# Pare-feu : seuls SSH, l'UI et l'API sont ouverts
resource "aws_security_group" "sg" {
  name        = "${var.name}-sg"
  description = "FoodVision : SSH, UI et API"

  dynamic "ingress" {
    for_each = toset(var.open_ports)
    content {
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "vm" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.sg.id]
  user_data              = file("${path.module}/cloud-init.yaml")

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  tags = {
    Name    = "${var.name}-prod"
    Project = "mlops"
  }
}
