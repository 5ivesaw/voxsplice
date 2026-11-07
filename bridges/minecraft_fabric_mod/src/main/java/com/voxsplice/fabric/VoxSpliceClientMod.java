package com.voxsplice.fabric;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.message.v1.ClientSendMessageEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.text.Text;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;

/**
 * VoxSplice Client-Side Fabric Mod for Minecraft + Simple Voice Chat (SVC).
 *
 * Intercepts outgoing chat messages starting with ".m " (e.g. ".m hello im human"),
 * cancels the server chat packet so nothing appears in public text chat, and fires
 * a localhost UDP datagram (127.0.0.1:9999) to the background VoxSplice daemon.
 */
public class VoxSpliceClientMod implements ClientModInitializer {
    private static final String PREFIX = ".m ";
    private static final String DAEMON_HOST = "127.0.0.1";
    private static final int DAEMON_PORT = 9999;

    @Override
    public void onInitializeClient() {
        ClientSendMessageEvents.ALLOW_CHAT.register((message) -> {
            if (message != null && message.startsWith(PREFIX)) {
                String phrase = message.substring(PREFIX.length()).trim();
                if (!phrase.isEmpty()) {
                    sendUdpToVoxSplice(phrase);
                    MinecraftClient client = MinecraftClient.getInstance();
                    if (client.player != null) {
                        client.player.sendMessage(
                            Text.literal("§8[§bVoxSplice§8] §7Splicing: §f\"" + phrase + "\""),
                            true
                        );
                    }
                }
                // Cancel packet so the ".m" command is never sent to public server chat
                return false;
            }
            return true;
        });
    }

    private void sendUdpToVoxSplice(String phrase) {
        new Thread(() -> {
            try (DatagramSocket socket = new DatagramSocket()) {
                String escaped = phrase.replace("\\", "\\\\").replace("\"", "\\\"");
                String json = "{\"text\":\"" + escaped + "\"}";
                byte[] buf = json.getBytes(StandardCharsets.UTF_8);
                InetAddress address = InetAddress.getByName(DAEMON_HOST);
                DatagramPacket packet = new DatagramPacket(buf, buf.length, address, DAEMON_PORT);
                socket.send(packet);
            } catch (Exception ignored) {
            }
        }, "VoxSplice-UDP-Sender").start();
    }
}
