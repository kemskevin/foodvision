variable "name" {
  type    = string
  default = "foodvision"
}

variable "region" {
  description = "eu-west-3 = Paris"
  type        = string
  default     = "eu-west-3"
}

variable "instance_type" {
  description = "m7i-flex.large (2 vCPU / 8 Go) : éligible au Free Tier pour les comptes créés après le 15/07/2025"
  type        = string
  default     = "m7i-flex.large"
}

variable "ssh_public_key_path" {
  type    = string
  default = "~/.ssh/id_ed25519.pub"
}

variable "open_ports" {
  description = "22 = SSH, 80 = UI + rapports drift, 8080 = API"
  type        = list(number)
  default     = [22, 80, 8080]
}
