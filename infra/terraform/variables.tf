variable "name" {
  type    = string
  default = "foodvision"
}

variable "zone" {
  type    = string
  default = "fr-par-1"
}

variable "region" {
  type    = string
  default = "fr-par"
}

variable "instance_type" {
  description = "DEV1-M (3 vCPU / 4 Go) suffit ; DEV1-L (4 vCPU / 8 Go) pour un test de charge plus confortable"
  type        = string
  default     = "DEV1-M"
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
