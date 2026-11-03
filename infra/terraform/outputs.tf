output "public_ip" {
  value = scaleway_instance_ip.public_ip.address
}

output "ssh" {
  value = "ssh root@${scaleway_instance_ip.public_ip.address}"
}

output "api_url" {
  value = "http://${scaleway_instance_ip.public_ip.address}:8080"
}

output "ui_url" {
  value = "http://${scaleway_instance_ip.public_ip.address}"
}
