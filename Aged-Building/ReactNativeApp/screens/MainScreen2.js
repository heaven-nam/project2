import React from 'react';
import {StyleSheet, Text, View, ScrollView} from 'react-native';

const LotsOfStyles = () => {
  return (
    <ScrollView>
      <View style={styles.container}>
        <Text style={styles.red}>안녕하세요</Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
    container: {
      marginTop: 200,
    },
    red: {
      color:'red',
      fontSize: 40,
    }
});

export default LotsOfStyles;