terraform {
  required_version = ">= 1.5"
  required_providers {
    scaleway = {
      source  = "scaleway/scaleway"
      version = "~> 2.40"
    }
  }
}

# Identifiants lus dans les variables d'environnement :
# SCW_ACCESS_KEY, SCW_SECRET_KEY, SCW_DEFAULT_PROJECT_ID, SCW_DEFAULT_ORGANIZATION_ID
provider "scaleway" {
  zone   = var.zone
  region = var.region
}

resource "scaleway_iam_ssh_key" "deployer" {
  name       = "${var.name}-deployer"
  public_key = file(pathexpand(var.ssh_public_key_path))
}

resource "scaleway_instance_ip" "public_ip" {}

resource "scaleway_instance_security_group" "sg" {
  name                    = "${var.name}-sg"
  inbound_default_policy  = "drop"
  outbound_default_policy = "accept"

  dynamic "inbound_rule" {
    for_each = toset(var.open_ports)
    content {
      action   = "accept"
      protocol = "TCP"
      port     = inbound_rule.value
    }
  }
}

resource "scaleway_instance_server" "vm" {
  name              = "${var.name}-prod"
  type              = var.instance_type
  image             = "ubuntu_jammy"
  ip_id             = scaleway_instance_ip.public_ip.id
  security_group_id = scaleway_instance_security_group.sg.id
  tags              = [var.name, "mlops"]

  user_data = {
    cloud-init = file("${path.module}/cloud-init.yaml")
  }

  depends_on = [scaleway_iam_ssh_key.deployer]
}
